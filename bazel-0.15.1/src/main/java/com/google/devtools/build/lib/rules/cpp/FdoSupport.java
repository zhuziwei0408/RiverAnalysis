// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.AnalysisEnvironment;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.skyframe.FileValue;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import com.google.devtools.build.skyframe.SkyFunction;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Collection;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;

/**
 * Support class for FDO (feedback directed optimization) and LIPO (lightweight inter-procedural
 * optimization).
 *
 * <p>Here follows a quick run-down of how FDO/LIPO builds work (for non-FDO/LIPO builds, none of
 * this applies):
 *
 * <p>{@link FdoSupport#create} is called from {@link FdoSupportFunction} (a {@link SkyFunction}),
 * which is requested from Skyframe by the {@code cc_toolchain} rule. It extracts the FDO .zip (in
 * case we work with an explicitly generated FDO profile file) or analyzes the .afdo.imports file
 * next to the .afdo file (if AutoFDO is in effect).
 *
 * <p>.afdo.imports files contain one import a line. A line is two paths separated by a colon, with
 * functions in the second path being referenced by functions in the first path. These are then put
 * into the imports map. If we do AutoFDO, we don't handle individual .gcda files, so gcdaFiles will
 * be empty.
 *
 * <p>Regular .fdo zip files contain .gcda files (which are added to gcdaFiles) and .gcda.imports
 * files. There is one .gcda.imports file for every source file and it contains one path in every
 * line, which can either be a path to a source file that contains a function referenced by the
 * original source file or the .gcda file for such a referenced file. They both are added to the
 * imports map.
 *
 * <p>If we do LIPO, we create an extra configuration that is called the "LIPO context collector",
 * whose job it is to collect information that every configured target compiled with LIPO needs. The
 * top-level target of this configuration is the LIPO context (always a cc_binary) and is an
 * implicit dependency of every cc_* rule through their :lipo_context_collector attribute. The
 * collected information is encapsulated in {@link LipoContextProvider}.
 *
 * <p>Note that the LIPO context can be different from the actual binary we are compiling because
 * it's beneficial to compile sources in a test in the exact same way as they would be compiled for
 * a particular {@code cc_binary} so that the code tested is the same as the one being run in
 * production. Thus, the {@code --lipo_context} command line flag, which takes the label of a {@code
 * cc_binary} rule as an argument which will be used as the LIPO context.
 *
 * <p>In this case, it can happen that files are needed for the compilation (because code in them is
 * inlined) that are not in the transitive closure of the tests being run. To cover this case, we
 * have the otherwise unused {@code :lipo_context} attribute, which depends on the LIPO context
 * without any configuration transition. Its purpose is to give a chance for the configured targets
 * containing the inlined code to run and thus create generating actions for the artifacts {@link
 * LipoContextProvider} contains. That is, configured targets in the LIPO context collector
 * configuration collect these artifacts but do not generate actions for them, and configured
 * targets under {@code :lipo_context} generate actions, but the artifacts they create are
 * discarded. This works because {@link Artifact} is a value object and the artifacts in {@link
 * LipoContextProvider} are {@code #equals()} to the ones created under {@code :lipo_context}.
 *
 * <p>For each C++ compile action in the target configuration, {@link #configureCompilation} is
 * called, which adds command line options and input files required for the build. There are three
 * cases:
 *
 * <ul>
 *   <li>If we do AutoFDO, the .afdo file and the source files containing the functions imported by
 *       the original source file (as determined from the inputs map) are added.
 *   <li>If we do FDO, the .gcda file corresponding to the source file is added.
 *   <li>If we do LIPO, in addition to the .gcda file corresponding to the source file (like for
 *       FDO) the source files that contain the functions referenced by the source file and their
 *       .gcda files are added, too.
 * </ul>
 *
 * <p>If we do LIPO, the actual {@code CcCompilationContext} for LIPO compilation actions is pieced
 * together from the {@code CcCompilationContext} in LipoContextProvider and that of the rule being
 * compiled. (see {@link CcCompilationContext#mergeForLipo}) This is so that the include files for
 * the extra LIPO sources are found and is, strictly speaking, incorrect, since it also changes the
 * declared include directories of the main source file, which in theory can result in the
 * compilation passing even though it should fail with undeclared inclusion errors.
 *
 * <p>During the actual execution of the C++ compile action, the extra sources also need to be
 * include scanned, which is the reason why they are {@link IncludeScannable} objects and not simple
 * artifacts. We currently create these {@link IncludeScannable} objects by creating actual C++
 * compile actions in the LIPO context collector configuration which are then never executed. In
 * fact, these C++ compile actions are never even registered with Skyframe. For this we propagate a
 * bit from {@code BuildConfiguration.isActionsEnabled} to {@code
 * CachingAnalysisEnvironment.allowRegisteringActions}, which causes actions to be silently
 * discarded after configured targets are created.
 */
@Immutable
@AutoCodec
public class FdoSupport {
  /**
   * The FDO mode we are operating in.
   *
   * LIPO can only be active if this is not <code>OFF</code>, but all of the modes below can work
   * with LIPO either off or on.
   */
  @VisibleForSerialization
  enum FdoMode {
    /** FDO is turned off. */
    OFF,

    /** Profiling-based FDO using an explicitly recorded profile. */
    VANILLA,

    /** FDO based on automatically collected data. */
    AUTO_FDO,

    /** Instrumentation-based FDO implemented on LLVM. */
    LLVM_FDO,
  }

  /**
   * Returns true if the given fdoFile represents an AutoFdo profile.
   */
  public static final boolean isAutoFdo(String fdoFile) {
    return CppFileTypes.GCC_AUTO_PROFILE.matches(fdoFile);
  }

  /**
   * Coverage information output directory passed to {@code --fdo_instrument},
   * or {@code null} if FDO instrumentation is disabled.
   */
  private final String fdoInstrument;

  /**
   * Path of the profile file passed to {@code --fdo_optimize}, or
   * {@code null} if FDO optimization is disabled.  The profile file
   * can be a coverage ZIP or an AutoFDO feedback file.
   */
  // TODO(lberki): this should be a PathFragment
  private final Path fdoProfile;

  /**
   * Temporary directory to which the coverage ZIP file is extracted to (relative to the exec root),
   * or {@code null} if FDO optimization is disabled. This is used to create artifacts for the
   * extracted files.
   *
   * <p>Note that this root is intentionally not registered with the artifact factory.
   */
  private final ArtifactRoot fdoRoot;

  /**
   * The relative path of the FDO root to the exec root.
   */
  private final PathFragment fdoRootExecPath;

  /**
   * Path of FDO files under the FDO root.
   */
  private final PathFragment fdoPath;

  /**
   * LIPO mode passed to {@code --lipo}. This is only used if
   * {@code fdoProfile != null}.
   */
  private final LipoMode lipoMode;

  /**
   * FDO mode.
   */
  private final FdoMode fdoMode;

  /**
   * The {@code .gcda} files that have been extracted from the ZIP file,
   * relative to the root of the ZIP file.
   */
  private final ImmutableSet<PathFragment> gcdaFiles;

  /**
   * Multimap from .gcda file base names to auxiliary input files.
   *
   * <p>The keys of the multimap are the exec root relative paths of .gcda files
   * with the extension removed. The values are the lines from the accompanying
   * .gcda.imports file.
   *
   * <p>The contents of the multimap are copied verbatim from the .gcda.imports
   * files and not yet checked for validity.
   */
  private final ImmutableMultimap<PathFragment, PathFragment> imports;

  /**
   * Creates an FDO support object.
   *
   * @param fdoInstrument value of the --fdo_instrument option
   * @param fdoProfile path to the profile file passed to --fdo_optimize option
   * @param lipoMode value of the --lipo_mode option
   */
  @VisibleForSerialization
  @AutoCodec.Instantiator
  FdoSupport(
      FdoMode fdoMode,
      LipoMode lipoMode,
      ArtifactRoot fdoRoot,
      PathFragment fdoRootExecPath,
      String fdoInstrument,
      Path fdoProfile,
      FdoZipContents fdoZipContents) {
    this.fdoInstrument = fdoInstrument;
    this.fdoProfile = fdoProfile;
    this.fdoRoot = fdoRoot;
    this.fdoRootExecPath = fdoRootExecPath;
    this.fdoPath = fdoProfile == null
        ? null
        : FileSystemUtils.removeExtension(PathFragment.create("_fdo").getChild(
            fdoProfile.getBaseName()));
    this.lipoMode = lipoMode;
    this.fdoMode = fdoMode;
    if (fdoZipContents != null) {
      this.gcdaFiles = fdoZipContents.gcdaFiles;
      this.imports = fdoZipContents.imports;
    } else {
      this.gcdaFiles = null;
      this.imports = null;
    }
  }

  public ArtifactRoot getFdoRoot() {
    return fdoRoot;
  }

  public Path getFdoProfile() {
    return fdoProfile;
  }

  @VisibleForSerialization
  // This method only exists for serialization.
  FdoZipContents getFdoZipContents() {
      return gcdaFiles == null ? null : new FdoZipContents(gcdaFiles, imports);
  }

  /** Creates an initialized {@link FdoSupport} instance. */
  static FdoSupport create(
      SkyFunction.Environment env,
      String fdoInstrument,
      Path fdoProfile,
      LipoMode lipoMode,
      Path execRoot,
      String productName,
      FdoMode fdoMode)
      throws IOException, FdoException, InterruptedException {

    if (fdoProfile == null) {
      lipoMode = LipoMode.OFF;
    }

    ArtifactRoot fdoRoot =
        (fdoProfile == null)
            ? null
            : ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative(productName + "-fdo"));

    PathFragment fdoRootExecPath = fdoProfile == null
        ? null
        : fdoRoot.getExecPath().getRelative(FileSystemUtils.removeExtension(
            PathFragment.create("_fdo").getChild(fdoProfile.getBaseName())));

    if (fdoProfile != null) {
      if (lipoMode != LipoMode.OFF) {
        // Incrementality is not supported for LIPO builds, see FdoSupport#scannables.
        // Ensure that the Skyframe value containing the configuration will not be reused to avoid
        // incrementality issues.
        PrecomputedValue.dependOnBuildId(env);
      } else {
        Path path = fdoMode == FdoMode.AUTO_FDO ? getAutoFdoImportsPath(fdoProfile) : fdoProfile;
        env.getValue(
            FileValue.key(
                RootedPath.toRootedPathMaybeUnderRoot(
                    path, ImmutableList.of(Root.fromPath(execRoot)))));
      }
    }

    if (env.valuesMissing()) {
      return null;
    }

    if (fdoMode == FdoMode.LLVM_FDO) {
      return new FdoSupport(
          fdoMode, LipoMode.OFF, fdoRoot, fdoRootExecPath, fdoInstrument, fdoProfile, null);
    }

    FdoZipContents fdoZipContents =
        extractFdoZip(fdoMode, lipoMode, execRoot, fdoProfile, fdoRootExecPath, productName);
    return new FdoSupport(
        fdoMode, lipoMode, fdoRoot, fdoRootExecPath, fdoInstrument, fdoProfile, fdoZipContents);
  }

  @Immutable
  @AutoCodec
  static class FdoZipContents {
    public static final ObjectCodec<FdoZipContents> CODEC =
        new FdoSupport_FdoZipContents_AutoCodec();

    private final ImmutableSet<PathFragment> gcdaFiles;
    private final ImmutableMultimap<PathFragment, PathFragment> imports;

    @VisibleForSerialization
    @AutoCodec.Instantiator
    FdoZipContents(ImmutableSet<PathFragment> gcdaFiles,
        ImmutableMultimap<PathFragment, PathFragment> imports) {
      this.gcdaFiles = gcdaFiles;
      this.imports = imports;
    }
  }

  /**
   * Extracts the FDO zip file and collects data from it that's needed during analysis.
   *
   * <p>When an {@code --fdo_optimize} compile is requested, unpacks the given
   * FDO gcda zip file into a clean working directory under execRoot.
   *
   * @throws FdoException if the FDO ZIP contains a file of unknown type
   */
  private static FdoZipContents extractFdoZip(FdoMode fdoMode, LipoMode lipoMode, Path execRoot,
      Path fdoProfile, PathFragment fdoRootExecPath, String productName)
      throws IOException, FdoException {
    // The execRoot != null case is only there for testing. We cannot provide a real ZIP file in
    // tests because ZipFileSystem does not work with a ZIP on an in-memory file system.
    // IMPORTANT: Keep in sync with #declareSkyframeDependencies to avoid incrementality issues.
    ImmutableSet<PathFragment> gcdaFiles = ImmutableSet.of();
    ImmutableMultimap<PathFragment, PathFragment> imports = ImmutableMultimap.of();

    if (fdoProfile != null && execRoot != null) {
      Path fdoDirPath = execRoot.getRelative(fdoRootExecPath);

      FileSystemUtils.deleteTreesBelow(fdoDirPath);
      FileSystemUtils.createDirectoryAndParents(fdoDirPath);

      if (fdoMode == FdoMode.AUTO_FDO) {
        if (lipoMode != LipoMode.OFF) {
          imports = readAutoFdoImports(getAutoFdoImportsPath(fdoProfile));
        }
        FileSystemUtils.ensureSymbolicLink(
            execRoot.getRelative(getAutoProfilePath(fdoProfile, fdoRootExecPath)), fdoProfile);
      } else {
        // Path objects referring to inside the zip file are only valid within this try block.
        // FdoZipContents doesn't reference any of them, so we are fine.
        if (!fdoProfile.exists()) {
          throw new FileNotFoundException(String.format("File '%s' does not exist", fdoProfile));
        }
        File zipIoFile = fdoProfile.getPathFile();
        File tempFile = null;
        try {
          // If it doesn't exist, we're dealing with an in-memory fs for testing
          // Copy the file and delete later
          if (!zipIoFile.exists()) {
            tempFile = File.createTempFile("bazel.test.", ".tmp");
            zipIoFile = tempFile;
            byte[] contents = FileSystemUtils.readContent(fdoProfile);
            try (OutputStream os = new FileOutputStream(tempFile)) {
              os.write(contents);
            }
          }
          try (ZipFile zipFile = new ZipFile(zipIoFile)) {
            String outputSymlinkName = productName + "-out";
            ImmutableSet.Builder<PathFragment> gcdaFilesBuilder = ImmutableSet.builder();
            ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder =
                ImmutableMultimap.builder();
            extractFdoZipDirectory(
                zipFile, fdoDirPath, outputSymlinkName, gcdaFilesBuilder, importsBuilder);
            gcdaFiles = gcdaFilesBuilder.build();
            imports = importsBuilder.build();
          }
        } finally {
          if (tempFile != null) {
            tempFile.delete();
          }
        }
      }
    }

    return new FdoZipContents(gcdaFiles, imports);
  }

  /**
   * Recursively extracts a directory from the GCDA ZIP file into a target directory.
   *
   * <p>Imports files are not written to disk. Their content is directly added to an internal data
   * structure.
   *
   * <p>The files are written at $EXECROOT/blaze-fdo/_fdo/(base name of profile zip), and the {@code
   * _fdo} directory there is symlinked to from the exec root, so that the file are also available
   * at $EXECROOT/_fdo/..., which is their exec path. We need to jump through these hoops because
   * the FDO root 1. needs to be a source root, thus the exec path of its root is ".", 2. it must
   * not be equal to the exec root so that the artifact factory does not get confused, 3. the files
   * under it must be reachable by their exec path from the exec root.
   *
   * @throws IOException if any of the I/O operations failed
   * @throws FdoException if the FDO ZIP contains a file of unknown type
   */
  private static void extractFdoZipDirectory(
      ZipFile zipFile,
      Path targetDir,
      String outputSymlinkName,
      ImmutableSet.Builder<PathFragment> gcdaFilesBuilder,
      ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder)
      throws IOException, FdoException {
    Enumeration<? extends ZipEntry> zipEntries = zipFile.entries();
    while (zipEntries.hasMoreElements()) {
      ZipEntry zipEntry = zipEntries.nextElement();
      if (zipEntry.isDirectory()) {
        continue;
      }
      String name = zipEntry.getName();
      if (!name.startsWith(outputSymlinkName)) {
        throw new ZipException(
            "FDO zip files must be zipped directly above '"
                + outputSymlinkName
                + "' for the compiler to find the profile");
      }
      Path targetFile = targetDir.getRelative(name);
      FileSystemUtils.createDirectoryAndParents(targetFile.getParentDirectory());
      if (CppFileTypes.COVERAGE_DATA.matches(name)) {
        try (OutputStream out = targetFile.getOutputStream()) {
          new ZipByteSource(zipFile, zipEntry).copyTo(out);
        }
        targetFile.setLastModifiedTime(zipEntry.getTime());
        targetFile.setWritable(false);
        targetFile.setExecutable(false);
        gcdaFilesBuilder.add(PathFragment.create(name));
      } else if (CppFileTypes.COVERAGE_DATA_IMPORTS.matches(name)) {
        readCoverageImports(zipFile, zipEntry, importsBuilder);
      }
    }
  }

  private static class ZipByteSource extends ByteSource {
    final ZipFile zipFile;
    final ZipEntry zipEntry;

    ZipByteSource(ZipFile zipFile, ZipEntry zipEntry) {
      this.zipFile = zipFile;
      this.zipEntry = zipEntry;
    }

    @Override
    public InputStream openStream() throws IOException {
      return zipFile.getInputStream(zipEntry);
    }
  }

  /**
   * Reads a .gcda.imports file and stores the imports information.
   *
   * @throws FdoException if an auxiliary LIPO input was not found
   */
  private static void readCoverageImports(
      ZipFile zipFile,
      ZipEntry importsFile,
      ImmutableMultimap.Builder<PathFragment, PathFragment> importsBuilder)
      throws IOException, FdoException {
    PathFragment key = PathFragment.create(importsFile.getName());
    String baseName = key.getBaseName();
    String ext = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA_IMPORTS.getExtensions());
    key = key.replaceName(baseName.substring(0, baseName.length() - ext.length()));
    for (String line :
        new ZipByteSource(zipFile, importsFile).asCharSource(ISO_8859_1).readLines()) {
      if (!line.isEmpty()) {
        // We can't yet fully check the validity of a line. this is done later
        // when we actually parse the contained paths.
        PathFragment execPath = PathFragment.create(line);
        if (execPath.isAbsolute()) {
          throw new FdoException(
              "Absolute paths not allowed in gcda imports file " + importsFile + ": " + execPath);
        }
        importsBuilder.put(key, PathFragment.create(line));
      }
    }
  }

  /**
   * Reads a .afdo.imports file and stores the imports information.
   */
  private static ImmutableMultimap<PathFragment, PathFragment> readAutoFdoImports(Path importsFile)
          throws IOException, FdoException {
    ImmutableMultimap.Builder<PathFragment, PathFragment> importBuilder =
        ImmutableMultimap.builder();
    for (String line : FileSystemUtils.iterateLinesAsLatin1(importsFile)) {
      line = line.trim();
      if (line.isEmpty()) {
        continue;
      }

      int colonIndex = line.indexOf(':');
      if (colonIndex < 0) {
        continue;
      }
      PathFragment key = PathFragment.create(line.substring(0, colonIndex));
      if (key.isAbsolute()) {
        throw new FdoException("Absolute paths not allowed in afdo imports file " + importsFile
            + ": " + key);
      }
      for (String auxFile : line.substring(colonIndex + 1).split(" ")) {
        if (auxFile.length() == 0) {
          continue;
        }

        importBuilder.put(key, PathFragment.create(auxFile));
      }
    }
    return importBuilder.build();
  }

  private static Path getAutoFdoImportsPath(Path fdoProfile) {
     return fdoProfile.getParentDirectory().getRelative(fdoProfile.getBaseName() + ".imports");
  }

  /**
   * Returns the imports from the .afdo.imports file of a source file.
   *
   * @param sourceExecPath the source file
   */
  private Collection<Artifact> getAutoFdoImports(RuleContext ruleContext,
      PathFragment sourceExecPath, LipoContextProvider lipoContextProvider) {
    Preconditions.checkState(lipoMode != LipoMode.OFF);
    ImmutableCollection<PathFragment> afdoImports = imports.get(sourceExecPath);
    Preconditions.checkState(afdoImports != null,
        "AutoFDO import data missing for %s", sourceExecPath);
    ImmutableList.Builder<Artifact> result = ImmutableList.builder();
    for (PathFragment afdoImport : afdoImports) {
      Artifact afdoArtifact = lipoContextProvider.getSourceArtifactMap().get(afdoImport);
      if (afdoArtifact != null) {
        result.add(afdoArtifact);
      } else {
        ruleContext.ruleError(String.format(
            "cannot find source file '%s' referenced from '%s'", afdoImport, sourceExecPath));
      }
    }
    return result.build();
  }

  /**
   * Returns the imports from the .gcda.imports file of an object file.
   *
   * @param objDirectory the object directory of the object file's target
   * @param objectName the object file
   */
  private Iterable<PathFragment> getImports(PathFragment objDirectory, PathFragment objectName) {
    Preconditions.checkState(lipoMode != LipoMode.OFF);
    Preconditions.checkState(imports != null,
        "Tried to look up imports of uninitialized FDOSupport");
    PathFragment key = objDirectory.getRelative(FileSystemUtils.removeExtension(objectName));
    ImmutableCollection<PathFragment> importsForObject = imports.get(key);
    Preconditions.checkState(importsForObject != null, "Import data missing for %s", key);
    return importsForObject;
  }

  /**
   * Configures a compile action builder by setting up command line options and auxiliary inputs
   * according to the FDO configuration. This method does nothing If FDO is disabled.
   */
  @ThreadSafe
  public ImmutableMap<String, String> configureCompilation(
      CppCompileActionBuilder builder,
      RuleContext ruleContext,
      PathFragment sourceName,
      PathFragment sourceExecPath,
      PathFragment outputName,
      boolean usePic,
      FeatureConfiguration featureConfiguration,
      FdoSupportProvider fdoSupportProvider) {

    ImmutableMap.Builder<String, String> variablesBuilder = ImmutableMap.builder();

    if ((fdoSupportProvider != null)
        && (fdoSupportProvider.getPrefetchHintsArtifact() != null)) {
      variablesBuilder.put(
          CompileBuildVariables.FDO_PREFETCH_HINTS_PATH.getVariableName(),
          fdoSupportProvider.getPrefetchHintsArtifact().getExecPathString());
    }

    // FDO is disabled -> do nothing.
    if ((fdoInstrument == null) && (fdoRoot == null)) {
      return ImmutableMap.of();
    }

    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      variablesBuilder.put(
          CompileBuildVariables.FDO_INSTRUMENT_PATH.getVariableName(), fdoInstrument);
    }

    // Optimization phase
    if (fdoRoot != null) {
      AnalysisEnvironment env = ruleContext.getAnalysisEnvironment();
      // Declare dependency on contents of zip file.
      if (env.getSkyframeEnv().valuesMissing()) {
        return ImmutableMap.of();
      }
      Iterable<Artifact> auxiliaryInputs =
          getAuxiliaryInputs(
              ruleContext, sourceName, sourceExecPath, outputName, usePic, fdoSupportProvider);
      builder.addMandatoryInputs(auxiliaryInputs);
      if (!Iterables.isEmpty(auxiliaryInputs)) {
        if (featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)) {
          variablesBuilder.put(
              CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
              getAutoProfilePath(fdoProfile, fdoRootExecPath).getPathString());
        }
        if (featureConfiguration.isEnabled(CppRuleClasses.FDO_OPTIMIZE)) {
          if (fdoMode == FdoMode.LLVM_FDO) {
            variablesBuilder.put(
                CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
                fdoSupportProvider.getProfileArtifact().getExecPathString());
          } else {
            variablesBuilder.put(
                CompileBuildVariables.FDO_PROFILE_PATH.getVariableName(),
                fdoRootExecPath.getPathString());
          }
        }
      }
    }
    return variablesBuilder.build();
  }

  /** Returns the auxiliary files that need to be added to the {@link CppCompileAction}. */
  private Iterable<Artifact> getAuxiliaryInputs(
      RuleContext ruleContext,
      PathFragment sourceName,
      PathFragment sourceExecPath,
      PathFragment outputName,
      boolean usePic,
      FdoSupportProvider fdoSupportProvider) {
    CcToolchainProvider toolchain =
        CppHelper.getToolchainUsingDefaultCcToolchainAttribute(ruleContext);
    LipoContextProvider lipoContextProvider =
        toolchain.isLLVMCompiler() ? null : CppHelper.getLipoContextProvider(ruleContext);

    ImmutableSet.Builder<Artifact> auxiliaryInputs = ImmutableSet.builder();

    if (fdoSupportProvider.getPrefetchHintsArtifact() != null) {
      auxiliaryInputs.add(fdoSupportProvider.getPrefetchHintsArtifact());
    }
    // If --fdo_optimize was not specified, we don't have any additional inputs.
    if (fdoProfile == null) {
      return auxiliaryInputs.build();
    } else if (fdoMode == FdoMode.LLVM_FDO || fdoMode == FdoMode.AUTO_FDO) {
      auxiliaryInputs.add(fdoSupportProvider.getProfileArtifact());
      if (lipoContextProvider != null) {
        auxiliaryInputs.addAll(getAutoFdoImports(ruleContext, sourceExecPath, lipoContextProvider));
      }
      return auxiliaryInputs.build();
    } else {
      PathFragment objectName =
          FileSystemUtils.appendExtension(outputName, usePic ? ".pic.o" : ".o");

      Label lipoLabel = ruleContext.getLabel();
      auxiliaryInputs.addAll(
          getGcdaArtifactsForObjectFileName(
              ruleContext, fdoSupportProvider, objectName, lipoLabel));

      if (lipoContextProvider != null) {
        for (PathFragment importedFile : getImports(
            getNonLipoObjDir(ruleContext, lipoLabel), objectName)) {
          if (CppFileTypes.COVERAGE_DATA.matches(importedFile.getBaseName())) {
            Artifact gcdaArtifact =
                getGcdaArtifactsForGcdaPath(fdoSupportProvider, importedFile);
            if (gcdaArtifact == null) {
              ruleContext.ruleError(String.format(
                  ".gcda file %s is not in the FDO zip (referenced by source file %s). Check if "
                  + "your profile is generated from the same sources you are building the "
                  + "optimized binary from",
                  importedFile, sourceName));
            } else {
              auxiliaryInputs.add(gcdaArtifact);
            }
          } else {
            Artifact importedArtifact = lipoContextProvider.getSourceArtifactMap()
                .get(importedFile);
            if (importedArtifact != null) {
              auxiliaryInputs.add(importedArtifact);
            } else {
              ruleContext.ruleError(String.format(
                  "cannot find source file '%s' referenced from '%s' by LIPO inclusion. Check if "
                  + "your profile is generated from the same sources you are building the "
                  + "optimized binary from",
                  importedFile, objectName));
            }
          }
        }
      }

      return auxiliaryInputs.build();
    }
  }

  /**
   * Returns the .gcda file artifacts for a .gcda path from the .gcda.imports file or null if the
   * referenced .gcda file is not in the FDO zip.
   */
  private Artifact getGcdaArtifactsForGcdaPath(FdoSupportProvider fdoSupportProvider,
      PathFragment gcdaPath) {
    if (!gcdaFiles.contains(gcdaPath)) {
      return null;
    }

    return fdoSupportProvider.getGcdaArtifacts().get(gcdaPath);
  }

  private PathFragment getNonLipoObjDir(RuleContext ruleContext, Label label) {
    return ruleContext.getConfiguration().getBinFragment()
        .getRelative(CppHelper.getObjDirectory(label));
  }

  /**
   * Returns a list of .gcda file artifacts for an object file path.
   *
   * <p>The resulting set is either empty (because no .gcda file exists for the
   * given object file) or contains one or two artifacts (the file itself and a
   * symlink to it).
   */
  private ImmutableList<Artifact> getGcdaArtifactsForObjectFileName(RuleContext ruleContext,
      FdoSupportProvider fdoSupportProvider, PathFragment objectFileName, Label lipoLabel) {
    // We put the .gcda files relative to the location of the .o file in the instrumentation run.
    String gcdaExt = Iterables.getOnlyElement(CppFileTypes.COVERAGE_DATA.getExtensions());
    PathFragment baseName = FileSystemUtils.replaceExtension(objectFileName, gcdaExt);
    PathFragment gcdaFile = getNonLipoObjDir(ruleContext, lipoLabel).getRelative(baseName);

    if (!gcdaFiles.contains(gcdaFile)) {
      // If the object is a .pic.o file and .pic.gcda is not found, we should try finding .gcda too
      String picoExt = Iterables.getOnlyElement(CppFileTypes.PIC_OBJECT_FILE.getExtensions());
      baseName = FileSystemUtils.replaceExtension(objectFileName, gcdaExt, picoExt);
      if (baseName == null) {
        // Object file is not .pic.o
        return ImmutableList.of();
      }
      gcdaFile = getNonLipoObjDir(ruleContext, lipoLabel).getRelative(baseName);
      if (!gcdaFiles.contains(gcdaFile)) {
        // .gcda file not found
        return ImmutableList.of();
      }
    }

    return ImmutableList.of(fdoSupportProvider.getGcdaArtifacts().get(gcdaFile));
  }


  private static PathFragment getAutoProfilePath(Path fdoProfile, PathFragment fdoRootExecPath) {
    return fdoRootExecPath.getRelative(getAutoProfileRootRelativePath(fdoProfile));
  }

  private static PathFragment getAutoProfileRootRelativePath(Path fdoProfile) {
    return PathFragment.create(fdoProfile.getBaseName());
  }

  /**
   * Returns whether AutoFDO is enabled.
   */
  @ThreadSafe
  public boolean isAutoFdoEnabled() {
    return fdoMode == FdoMode.AUTO_FDO;
  }

  /**
   * Adds the FDO profile output path to the variable builder. If FDO is disabled, no build variable
   * is added.
   */
  @ThreadSafe
  public void getLinkOptions(
      FeatureConfiguration featureConfiguration, CcToolchainVariables.Builder buildVariables) {
    if (featureConfiguration.isEnabled(CppRuleClasses.FDO_INSTRUMENT)) {
      buildVariables.addStringVariable("fdo_instrument_path", fdoInstrument);
    }
  }

  /**
   * Adds the AutoFDO profile path to the variable builder and returns the profile artifact. If
   * AutoFDO is disabled, no build variable is added and returns null.
   */
  @ThreadSafe
  public ProfileArtifacts buildProfileForLtoBackend(
      FdoSupportProvider fdoSupportProvider,
      FeatureConfiguration featureConfiguration,
      CcToolchainVariables.Builder buildVariables,
      RuleContext ruleContext) {
    Artifact prefetch = fdoSupportProvider.getPrefetchHintsArtifact();
    if (prefetch != null) {
      buildVariables.addStringVariable("fdo_prefetch_hints_path", prefetch.getExecPathString());
    }
    if (!featureConfiguration.isEnabled(CppRuleClasses.AUTOFDO)) {
      return new ProfileArtifacts(null, prefetch);
    }

    Artifact profile = fdoSupportProvider.getProfileArtifact();
    buildVariables.addStringVariable("fdo_profile_path", profile.getExecPathString());
    return new ProfileArtifacts(profile, prefetch);
  }

  /**
   * Returns the path of the FDO output tree (relative to the execution root)
   * containing the .gcda profile files, or null if FDO is not enabled.
   */
  @VisibleForTesting
  public PathFragment getFdoOptimizeDir() {
    return fdoRootExecPath;
  }

  public FdoSupportProvider createFdoSupportProvider(
      RuleContext ruleContext, ProfileArtifacts profiles) {
    if (fdoRoot == null) {
      return new FdoSupportProvider(this, profiles, null);
    }

    if (fdoMode == FdoMode.LLVM_FDO) {
      Preconditions.checkState(profiles != null && profiles.getProfileArtifact() != null);
      return new FdoSupportProvider(this, profiles, null);
    }

    Preconditions.checkState(fdoPath != null);
    PathFragment profileRootRelativePath = getAutoProfileRootRelativePath(fdoProfile);

    Artifact profileArtifact =
        ruleContext
            .getAnalysisEnvironment()
            .getDerivedArtifact(fdoPath.getRelative(profileRootRelativePath), fdoRoot);
    ruleContext.registerAction(new FdoStubAction(ruleContext.getActionOwner(), profileArtifact));
    Preconditions.checkState(fdoPath != null);
    ImmutableMap.Builder<PathFragment, Artifact> gcdaArtifacts = ImmutableMap.builder();
    for (PathFragment path : gcdaFiles) {
      Artifact gcdaArtifact = ruleContext.getAnalysisEnvironment().getDerivedArtifact(
          fdoPath.getRelative(path), fdoRoot);
      ruleContext.registerAction(new FdoStubAction(ruleContext.getActionOwner(), gcdaArtifact));
      gcdaArtifacts.put(path, gcdaArtifact);
    }

    return new FdoSupportProvider(
        this,
        new ProfileArtifacts(
            profileArtifact, profiles == null ? null : profiles.getPrefetchHintsArtifact()),
        gcdaArtifacts.build());
  }

  /**
   * An exception indicating an issue with FDO coverage files.
   */
  public static final class FdoException extends Exception {
    FdoException(String message) {
      super(message);
    }
  }
}
