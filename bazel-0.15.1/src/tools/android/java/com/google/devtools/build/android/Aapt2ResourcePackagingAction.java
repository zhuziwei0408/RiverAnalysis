// Copyright 2015 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.android;

import static com.google.common.collect.Streams.concat;
import static java.util.stream.Collectors.toList;

import com.android.utils.StdLogger;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.android.AndroidResourceProcessingAction.Options;
import com.google.devtools.build.android.aapt2.Aapt2ConfigOptions;
import com.google.devtools.build.android.aapt2.CompiledResources;
import com.google.devtools.build.android.aapt2.PackagedResources;
import com.google.devtools.build.android.aapt2.ResourceCompiler;
import com.google.devtools.build.android.aapt2.ResourceLinker;
import com.google.devtools.build.android.aapt2.StaticLibrary;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.ShellQuotedParamsFilePreProcessor;
import com.google.devtools.common.options.TriState;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Provides an entry point for the resource processing using the AOSP build tools.
 *
 * <pre>
 * Example Usage:
 *   java/com/google/build/android/Aapt2ResourcePackagingAction\
 *      --sdkRoot path/to/sdk\
 *      --aapt path/to/sdk/aapt\
 *      --adb path/to/sdk/adb\
 *      --zipAlign path/to/sdk/zipAlign\
 *      --androidJar path/to/sdk/androidJar\
 *      --manifestOutput path/to/manifest\
 *      --primaryData path/to/resources:path/to/assets:path/to/manifest\
 *      --data p/t/res1:p/t/assets1:p/t/1/AndroidManifest.xml:p/t/1/R.txt:symbols,\
 *             p/t/res2:p/t/assets2:p/t/2/AndroidManifest.xml:p/t/2/R.txt:symbols\
 *      --packagePath path/to/write/archive.ap_\
 *      --srcJarOutput path/to/write/archive.srcjar
 * </pre>
 */
public class Aapt2ResourcePackagingAction {

  private static final StdLogger STD_LOGGER = new StdLogger(StdLogger.Level.WARNING);

  private static Aapt2ConfigOptions aaptConfigOptions;
  private static Options options;

  public static void main(String[] args) throws Exception {
    Profiler profiler = LoggingProfiler.createAndStart("setup");
    OptionsParser optionsParser =
        OptionsParser.newOptionsParser(Options.class, Aapt2ConfigOptions.class);
    optionsParser.enableParamsFileSupport(
        new ShellQuotedParamsFilePreProcessor(FileSystems.getDefault()));
    optionsParser.parseAndExitUponError(args);
    aaptConfigOptions = optionsParser.getOptions(Aapt2ConfigOptions.class);
    options = optionsParser.getOptions(Options.class);

    try (ScopedTemporaryDirectory scopedTmp =
            new ScopedTemporaryDirectory("android_resources_tmp");
        ExecutorServiceCloser executorService = ExecutorServiceCloser.createWithFixedPoolOf(15)) {
      final Path tmp = scopedTmp.getPath();
      final Path mergedAssets = tmp.resolve("merged_assets");
      final Path mergedResources = tmp.resolve("merged_resources");
      final Path filteredResources = tmp.resolve("filtered_resources");

      final Path densityManifest = tmp.resolve("manifest-filtered/AndroidManifest.xml");

      final Path processedManifest = tmp.resolve("manifest-processed/AndroidManifest.xml");
      final Path databindingResourcesRoot =
          Files.createDirectories(tmp.resolve("android_data_binding_resources"));
      final Path compiledResources = Files.createDirectories(tmp.resolve("compiled"));
      final Path linkedOut = Files.createDirectories(tmp.resolve("linked"));

      profiler.recordEndOf("setup").startTask("merging");

      AndroidDataDeserializer dataDeserializer =
          aaptConfigOptions.useCompiledResourcesForMerge
              ? AndroidCompiledDataDeserializer.withFilteredResources(options.prefilteredResources)
              : AndroidParsedDataDeserializer.withFilteredResources(options.prefilteredResources);

      // Checks for merge conflicts.
      MergedAndroidData mergedAndroidData =
          AndroidResourceMerger.mergeData(
                  ParsedAndroidData.from(options.primaryData),
                  options.primaryData.getManifest(),
                  ImmutableList.<SerializedAndroidData>builder()
                      .addAll(options.directData)
                      .addAll(options.directAssets)
                      .build(),
                  ImmutableList.<SerializedAndroidData>builder()
                      .addAll(options.transitiveData)
                      .addAll(options.transitiveAssets)
                      .build(),
                  mergedResources,
                  mergedAssets,
                  null /* cruncher. Aapt2 automatically chooses to crunch or not. */,
                  options.packageType,
                  options.symbolsOut,
                  /* rclassWriter= */ null,
                  dataDeserializer,
                  options.throwOnResourceConflict,
                  executorService)
              .filter(
                  new DensitySpecificResourceFilter(
                      options.densities, filteredResources, mergedResources),
                  new DensitySpecificManifestProcessor(options.densities, densityManifest));

      profiler.recordEndOf("merging");

     
        profiler.startTask("compile");
      final ResourceCompiler compiler =
          ResourceCompiler.create(
              executorService,
              compiledResources,
              aaptConfigOptions.aapt2,
              aaptConfigOptions.buildToolsVersion,
              aaptConfigOptions.generatePseudoLocale);

        CompiledResources compiled =
            options
                .primaryData
                .processDataBindings(
                    options.dataBindingInfoOut, options.packageForR, databindingResourcesRoot)
                .compile(compiler, compiledResources)
                .processManifest(
                    manifest ->
                        AndroidManifestProcessor.with(STD_LOGGER)
                            .processManifest(
                                options.applicationId,
                                options.versionCode,
                                options.versionName,
                                manifest,
                                processedManifest))
                .processManifest(
                    manifest ->
                        new DensitySpecificManifestProcessor(options.densities, densityManifest)
                            .process(manifest));
        profiler.recordEndOf("compile").startTask("link");
        // Write manifestOutput now before the dummy manifest is created.
        if (options.manifestOutput != null) {
          AndroidResourceOutputs.copyManifestToOutput(compiled, options.manifestOutput);
        }

        List<CompiledResources> compiledResourceDeps =
            // Last defined dependencies will overwrite previous one, so always place direct
            // after transitive.
            concat(options.transitiveData.stream(), options.directData.stream())
                .map(DependencyAndroidData::getCompiledSymbols)
                .collect(toList());

      List<Path> assetDirs =
          concat(
                  options.transitiveData.stream(),
                  options.transitiveAssets.stream(),
                  options.directData.stream(),
                  options.directAssets.stream())
              .flatMap(dep -> dep.assetDirs.stream())
              .collect(toList());
        assetDirs.addAll(options.primaryData.assetDirs);

      final PackagedResources packagedResources =
          ResourceLinker.create(aaptConfigOptions.aapt2, executorService, linkedOut)
              .profileUsing(profiler)
              .customPackage(options.packageForR)
              .outputAsProto(aaptConfigOptions.resourceTableAsProto)
              .dependencies(ImmutableList.of(StaticLibrary.from(aaptConfigOptions.androidJar)))
              .include(compiledResourceDeps)
              .withAssets(assetDirs)
              .buildVersion(aaptConfigOptions.buildToolsVersion)
              .conditionalKeepRules(aaptConfigOptions.conditionalKeepRules == TriState.YES)
              .filterToDensity(options.densities)
              .debug(aaptConfigOptions.debug)
              .includeGeneratedLocales(aaptConfigOptions.generatePseudoLocale)
              .includeOnlyConfigs(aaptConfigOptions.resourceConfigs)
              .link(compiled)
              .copyPackageTo(options.packagePath)
              .copyProguardTo(options.proguardOutput)
              .copyMainDexProguardTo(options.mainDexProguardOutput)
              .createSourceJar(options.srcJarOutput)
              .copyRTxtTo(options.rOutput);
        profiler.recordEndOf("link");
        if (options.resourcesOutput != null) {
          profiler.startTask("package");
        // The compiled resources and the merged resources should be the same.
        // TODO(corysmith): Decompile or otherwise provide the exact resources in the apk.
        ResourcesZip.fromApk(
                mergedAndroidData.getResourceDir(),
                packagedResources.getApk(),
                packagedResources.getResourceIds())
            .writeTo(options.resourcesOutput, /* compress= */ false);
          profiler.recordEndOf("package");
        }
      }
    }
  }
