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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FailAction;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.MakeVariableSupplier.MapBackedMakeVariableSupplier;
import com.google.devtools.build.lib.analysis.OutputGroupInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetFactory;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.rules.cpp.CcCommon.CcFlagsSupplier;
import com.google.devtools.build.lib.rules.cpp.CcCompilationHelper.CompilationInfo;
import com.google.devtools.build.lib.rules.cpp.CcLinkingHelper.LinkingInfo;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.List;

/**
 * A ConfiguredTarget for <code>cc_library</code> rules.
 */
public abstract class CcLibrary implements RuleConfiguredTargetFactory {

  private final CppSemantics semantics;

  protected CcLibrary(CppSemantics semantics) {
    this.semantics = semantics;
  }

  // These file extensions don't generate object files.
  private static final FileTypeSet NO_OBJECT_GENERATING_FILETYPES = FileTypeSet.of(
      CppFileTypes.CPP_HEADER, CppFileTypes.ARCHIVE, CppFileTypes.PIC_ARCHIVE,
      CppFileTypes.ALWAYS_LINK_LIBRARY, CppFileTypes.ALWAYS_LINK_PIC_LIBRARY,
      CppFileTypes.SHARED_LIBRARY, CppFileTypes.VERSIONED_SHARED_LIBRARY);

  private static Runfiles collectRunfiles(
      RuleContext context,
      FeatureConfiguration featureConfiguration,
      CcLinkingOutputs ccLinkingOutputs,
      CcToolchainProvider ccToolchain,
      boolean neverLink,
      boolean addDynamicRuntimeInputArtifactsToRunfiles,
      boolean linkingStatically) {
    Runfiles.Builder builder = new Runfiles.Builder(
        context.getWorkspaceName(), context.getConfiguration().legacyExternalRunfiles());

    // neverlink= true creates a library that will never be linked into any binary that depends on
    // it, but instead be loaded as an extension. So we need the dynamic library for this in the
    // runfiles.
    builder.addArtifacts(ccLinkingOutputs.getLibrariesForRunfiles(linkingStatically && !neverLink));
    builder.add(context, CcRunfiles.runfilesFunction(linkingStatically));

    builder.addDataDeps(context);

    if (addDynamicRuntimeInputArtifactsToRunfiles) {
      builder.addTransitiveArtifacts(ccToolchain.getDynamicRuntimeLinkInputs(featureConfiguration));
    }
    return builder.build();
  }

  @Override
  public ConfiguredTarget create(RuleContext context)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(context);
    boolean linkStatic = context.attributes().get("linkstatic", Type.BOOLEAN);
    init(
        semantics,
        context,
        builder,
        /* additionalCopts= */ ImmutableList.of(),
        context.attributes().get("alwayslink", Type.BOOLEAN),
        /* neverLink= */ false,
        linkStatic,
        /* addDynamicRuntimeInputArtifactsToRunfiles= */ false);
    return builder.build();
  }

  public static void init(
      CppSemantics semantics,
      RuleContext ruleContext,
      RuleConfiguredTargetBuilder targetBuilder,
      ImmutableList<String> additionalCopts,
      boolean alwaysLink,
      boolean neverLink,
      boolean linkStatic,
      boolean addDynamicRuntimeInputArtifactsToRunfiles)
      throws RuleErrorException, InterruptedException {
    final CcCommon common = new CcCommon(ruleContext);

    CcToolchainProvider ccToolchain = common.getToolchain();

    if (CppHelper.shouldUseToolchainForMakeVariables(ruleContext)) {
      ImmutableMap.Builder<String, String> toolchainMakeVariables = ImmutableMap.builder();
      ccToolchain.addGlobalMakeVariables(toolchainMakeVariables);
      ruleContext.initConfigurationMakeVariableContext(
          new MapBackedMakeVariableSupplier(toolchainMakeVariables.build()),
          new CcFlagsSupplier(ruleContext));
    } else {
      ruleContext.initConfigurationMakeVariableContext(new CcFlagsSupplier(ruleContext));
    }

    FdoSupportProvider fdoSupport = common.getFdoSupport();
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(ruleContext, ccToolchain);
    PrecompiledFiles precompiledFiles = new PrecompiledFiles(ruleContext);

    semantics.validateAttributes(ruleContext);
    if (ruleContext.hasErrors()) {
      return;
    }

    CcCompilationHelper compilationHelper =
        new CcCompilationHelper(
                ruleContext, semantics, featureConfiguration, ccToolchain, fdoSupport)
            .fromCommon(common, additionalCopts)
            .addSources(common.getSources())
            .addPrivateHeaders(common.getPrivateHeaders())
            .addPublicHeaders(common.getHeaders())
            .enableCompileProviders()
            .addPrecompiledFiles(precompiledFiles);

    CcLinkingHelper linkingHelper =
        new CcLinkingHelper(
                ruleContext,
                semantics,
                featureConfiguration,
                ccToolchain,
                fdoSupport,
                ruleContext.getConfiguration())
            .fromCommon(common)
            .addLinkopts(common.getLinkopts())
            .enableCcNativeLibrariesProvider()
            .enableInterfaceSharedObjects()
            // Generate .a and .so outputs even without object files to fulfill the rule class
            // contract wrt. implicit output files, if the contract says so. Behavior here differs
            // between Bazel and Blaze.
            .setGenerateLinkActionsIfEmpty(
                ruleContext.getRule().getImplicitOutputsFunction() != ImplicitOutputsFunction.NONE)
            .setAlwayslink(alwaysLink)
            .setNeverLink(neverLink)
            .addLinkstamps(ruleContext.getPrerequisites("linkstamp", Mode.TARGET));

    Artifact soImplArtifact = null;
    boolean supportsDynamicLinker = ccToolchain.supportsDynamicLinker();
    // TODO(djasper): This is hacky. We should actually try to figure out whether we generate
    // ccOutputs.
    boolean createDynamicLibrary =
        !linkStatic
            && supportsDynamicLinker
            && (appearsToHaveObjectFiles(ruleContext.attributes())
                || featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN));
    if (ruleContext.getRule().isAttrDefined("outs", Type.STRING_LIST)) {
      List<String> outs = ruleContext.attributes().get("outs", Type.STRING_LIST);
      if (outs.size() > 1) {
        ruleContext.attributeError("outs", "must be a singleton list");
      } else if (outs.size() == 1) {
        PathFragment soImplFilename = PathFragment.create(ruleContext.getLabel().getName());
        soImplFilename = soImplFilename.replaceName(outs.get(0));
        if (!soImplFilename.getPathString().endsWith(".so")) { // Sanity check.
          ruleContext.attributeError("outs", "file name must end in '.so'");
        }

        if (createDynamicLibrary) {
          soImplArtifact = ruleContext.getBinArtifact(soImplFilename);
        }
      }
    }

    if (ruleContext.getRule().isAttrDefined("srcs", BuildType.LABEL_LIST)) {
      ruleContext.checkSrcsSamePackage(true);
    }
    if (ruleContext.getRule().isAttrDefined("textual_hdrs", BuildType.LABEL_LIST)) {
      compilationHelper.addPublicTextualHeaders(
          ruleContext.getPrerequisiteArtifacts("textual_hdrs", Mode.TARGET).list());
    }

    if (common.getLinkopts().contains("-static")) {
      ruleContext.attributeWarning("linkopts", "Using '-static' here won't work. "
                                   + "Did you mean to use 'linkstatic=1' instead?");
    }

    linkingHelper.setShouldCreateDynamicLibrary(createDynamicLibrary);
    linkingHelper.setDynamicLibrary(soImplArtifact);

    // If the reason we're not creating a dynamic library is that the toolchain
    // doesn't support it, then register an action which complains when triggered,
    // which only happens when some rule explicitly depends on the dynamic library.
    if (!createDynamicLibrary && !supportsDynamicLinker) {
      ImmutableList.Builder<Artifact> dynamicLibraries = ImmutableList.builder();
      dynamicLibraries.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              ruleContext.getConfiguration(),
              LinkTargetType.NODEPS_DYNAMIC_LIBRARY));
      if (CppHelper.useInterfaceSharedObjects(ccToolchain.getCppConfiguration(), ccToolchain)) {
        dynamicLibraries.add(
            CppHelper.getLinkedArtifact(
                ruleContext,
                ccToolchain,
                ruleContext.getConfiguration(),
                LinkTargetType.INTERFACE_DYNAMIC_LIBRARY));
      }
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          dynamicLibraries.build(), "Toolchain does not support dynamic linking"));
    } else if (!createDynamicLibrary
        && ruleContext.attributes().isConfigurable("srcs")) {
      // If "srcs" is configurable, the .so output is always declared because the logic that
      // determines implicit outs doesn't know which value of "srcs" will ultimately get chosen.
      // Here, where we *do* have the correct value, it may not contain any source files to
      // generate an .so with. If that's the case, register a fake generating action to prevent
      // a "no generating action for this artifact" error.
      ImmutableList.Builder<Artifact> dynamicLibraries = ImmutableList.builder();
      dynamicLibraries.add(
          CppHelper.getLinkedArtifact(
              ruleContext,
              ccToolchain,
              ruleContext.getConfiguration(),
              LinkTargetType.NODEPS_DYNAMIC_LIBRARY));
      if (CppHelper.useInterfaceSharedObjects(ccToolchain.getCppConfiguration(), ccToolchain)) {
        dynamicLibraries.add(
            CppHelper.getLinkedArtifact(
                ruleContext,
                ccToolchain,
                ruleContext.getConfiguration(),
                LinkTargetType.INTERFACE_DYNAMIC_LIBRARY));
      }
      ruleContext.registerAction(new FailAction(ruleContext.getActionOwner(),
          dynamicLibraries.build(), "configurable \"srcs\" triggers an implicit .so output "
          + "even though there are no sources to compile in this configuration"));
    }

    /*
     * Add the libraries from srcs, if any. For static/mostly static
     * linking we setup the dynamic libraries if there are no static libraries
     * to choose from. Path to the libraries will be mangled to avoid using
     * absolute path names on the -rpath, but library filenames will be
     * preserved (since some libraries might have SONAME tag) - symlink will
     * be created to the parent directory instead.
     *
     * For compatibility with existing BUILD files, any ".a" or ".lo" files listed in
     * srcs are assumed to be position-independent code, or at least suitable for
     * inclusion in shared libraries, unless they end with ".nopic.a" or ".nopic.lo".
     *
     * Note that some target platforms do not require shared library code to be PIC.
     */
    Iterable<LibraryToLink> staticLibrariesFromSrcs = LinkerInputs.opaqueLibrariesToLink(
        ArtifactCategory.STATIC_LIBRARY, precompiledFiles.getStaticLibraries());
    Iterable<LibraryToLink> alwayslinkLibrariesFromSrcs = LinkerInputs.opaqueLibrariesToLink(
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY,
        precompiledFiles.getAlwayslinkStaticLibraries());
    Iterable<LibraryToLink> picStaticLibrariesFromSrcs = LinkerInputs.opaqueLibrariesToLink(
        ArtifactCategory.STATIC_LIBRARY, precompiledFiles.getPicStaticLibraries());
    Iterable<LibraryToLink> picAlwayslinkLibrariesFromSrcs = LinkerInputs.opaqueLibrariesToLink(
        ArtifactCategory.ALWAYSLINK_STATIC_LIBRARY, precompiledFiles.getPicAlwayslinkLibraries());

    linkingHelper.addStaticLibraries(staticLibrariesFromSrcs);
    linkingHelper.addStaticLibraries(alwayslinkLibrariesFromSrcs);
    linkingHelper.addPicStaticLibraries(picStaticLibrariesFromSrcs);
    linkingHelper.addPicStaticLibraries(picAlwayslinkLibrariesFromSrcs);
    Iterable<LibraryToLink> dynamicLibraries =
        Iterables.transform(
            precompiledFiles.getSharedLibraries(),
            library ->
                LinkerInputs.solibLibraryToLink(
                    common.getDynamicLibrarySymlink(library, true),
                    library,
                    CcLinkingOutputs.libraryIdentifierOf(library)));
    linkingHelper.addDynamicLibraries(dynamicLibraries);
    linkingHelper.addExecutionDynamicLibraries(dynamicLibraries);

    CompilationInfo compilationInfo = compilationHelper.compile();
    LinkingInfo linkingInfo =
        linkingHelper.link(
            compilationInfo.getCcCompilationOutputs(), compilationInfo.getCcCompilationContext());

    /*
     * We always generate a static library, even if there aren't any source files.
     * This keeps things simpler by avoiding special cases when making use of the library.
     * For example, this is needed to ensure that building a library with "bazel build"
     * will also build all of the library's "deps".
     * However, we only generate a dynamic library if there are source files.
     */
    // For now, we don't add the precompiled libraries to the files to build.
    CcLinkingOutputs linkedLibraries =
        linkingInfo.getCcLinkingOutputsExcludingPrecompiledLibraries();

    NestedSetBuilder<Artifact> filesBuilder = NestedSetBuilder.stableOrder();
    filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(linkedLibraries.getStaticLibraries()));
    filesBuilder.addAll(LinkerInputs.toLibraryArtifacts(linkedLibraries.getPicStaticLibraries()));

    if (!featureConfiguration.isEnabled(CppRuleClasses.TARGETS_WINDOWS)) {
      filesBuilder.addAll(
          LinkerInputs.toNonSolibArtifacts(linkedLibraries.getDynamicLibrariesForLinking()));
      filesBuilder.addAll(
          LinkerInputs.toNonSolibArtifacts(linkedLibraries.getDynamicLibrariesForRuntime()));
    }

    CcLinkingOutputs linkingOutputs = linkingInfo.getCcLinkingOutputs();
    if (!featureConfiguration.isEnabled(CppRuleClasses.HEADER_MODULE_CODEGEN)) {
      warnAboutEmptyLibraries(ruleContext, compilationInfo.getCcCompilationOutputs(), linkStatic);
    }
    NestedSet<Artifact> filesToBuild = filesBuilder.build();

    List<Artifact> instrumentedObjectFiles = new ArrayList<>();
    instrumentedObjectFiles.addAll(compilationInfo.getCcCompilationOutputs().getObjectFiles(false));
    instrumentedObjectFiles.addAll(compilationInfo.getCcCompilationOutputs().getObjectFiles(true));
    InstrumentedFilesProvider instrumentedFilesProvider =
        common.getInstrumentedFilesProvider(
            instrumentedObjectFiles, /* withBaselineCoverage= */ true);
    CppHelper.maybeAddStaticLinkMarkerProvider(targetBuilder, ruleContext);

    Runfiles staticRunfiles =
        collectRunfiles(
            ruleContext,
            featureConfiguration,
            linkingOutputs,
            ccToolchain,
            neverLink,
            addDynamicRuntimeInputArtifactsToRunfiles,
            true);
    Runfiles sharedRunfiles =
        collectRunfiles(
            ruleContext,
            featureConfiguration,
            linkingOutputs,
            ccToolchain,
            neverLink,
            addDynamicRuntimeInputArtifactsToRunfiles,
            false);

    targetBuilder
        .setFilesToBuild(filesToBuild)
        .addProviders(compilationInfo.getProviders())
        .addProviders(linkingInfo.getProviders())
        .addNativeDeclaredProvider(
            overrideRunfilesProvider(staticRunfiles, sharedRunfiles, linkingInfo))
        .addSkylarkTransitiveInfo(CcSkylarkApiProvider.NAME, new CcSkylarkApiProvider())
        .addOutputGroups(
            CcCommon.mergeOutputGroups(
                ImmutableList.of(compilationInfo.getOutputGroups(), linkingInfo.getOutputGroups())))
        .addProvider(InstrumentedFilesProvider.class, instrumentedFilesProvider)
        .addProvider(
            RunfilesProvider.class, RunfilesProvider.withData(staticRunfiles, sharedRunfiles))
        .addOutputGroup(
            OutputGroupInfo.HIDDEN_TOP_LEVEL,
            collectHiddenTopLevelArtifacts(
                ruleContext, ccToolchain, compilationInfo.getCcCompilationOutputs()))
        .addOutputGroup(
            CcCompilationHelper.HIDDEN_HEADER_TOKENS,
            CcCompilationHelper.collectHeaderTokens(
                ruleContext, compilationInfo.getCcCompilationOutputs()));
  }

  private static CcLinkingInfo overrideRunfilesProvider(
      Runfiles staticRunfiles, Runfiles sharedRunfiles, LinkingInfo linkingInfo) {
    CcLinkingInfo ccLinkingInfo =
        (CcLinkingInfo) linkingInfo.getProviders().getProvider(CcLinkingInfo.PROVIDER.getKey());
    CcLinkingInfo.Builder ccLinkingInfoBuilder = CcLinkingInfo.Builder.create();
    ccLinkingInfoBuilder.setCcLinkParamsStore(ccLinkingInfo.getCcLinkParamsStore());
    ccLinkingInfoBuilder.setCcExecutionDynamicLibraries(
        ccLinkingInfo.getCcExecutionDynamicLibraries());
    ccLinkingInfoBuilder.setCcRunfiles(new CcRunfiles(staticRunfiles, sharedRunfiles));

    return ccLinkingInfoBuilder.build();
  }

  private static NestedSet<Artifact> collectHiddenTopLevelArtifacts(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      CcCompilationOutputs ccCompilationOutputs) {
    // Ensure that we build all the dependencies, otherwise users may get confused.
    NestedSetBuilder<Artifact> artifactsToForceBuilder = NestedSetBuilder.stableOrder();
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    boolean isLipoCollector = cppConfiguration.isLipoContextCollector();
    boolean processHeadersInDependencies = cppConfiguration.processHeadersInDependencies();
    boolean usePic = CppHelper.usePicForDynamicLibraries(ruleContext, toolchain);
    artifactsToForceBuilder.addTransitive(
        ccCompilationOutputs.getFilesToCompile(
            isLipoCollector, processHeadersInDependencies, usePic));
    for (OutputGroupInfo dep :
        ruleContext.getPrerequisites(
            "deps", Mode.TARGET, OutputGroupInfo.SKYLARK_CONSTRUCTOR)) {
      artifactsToForceBuilder.addTransitive(
          dep.getOutputGroup(OutputGroupInfo.HIDDEN_TOP_LEVEL));
    }
    return artifactsToForceBuilder.build();
  }

  private static void warnAboutEmptyLibraries(RuleContext ruleContext,
      CcCompilationOutputs ccCompilationOutputs,
      boolean linkstaticAttribute) {
    if (ruleContext.getFragment(CppConfiguration.class).isLipoContextCollector()) {
      // Do not signal warnings in the lipo context collector configuration. These will be duly
      // signaled in the target configuration, and there can be spurious warnings since targets in
      // the LIPO context collector configuration do not compile anything.
      return;
    }
    if (ccCompilationOutputs.getObjectFiles(false).isEmpty()
        && ccCompilationOutputs.getObjectFiles(true).isEmpty()) {
      if (!linkstaticAttribute && appearsToHaveObjectFiles(ruleContext.attributes())) {
        ruleContext.attributeWarning("linkstatic",
            "setting 'linkstatic=1' is recommended if there are no object files");
      }
    } else {
      if (!linkstaticAttribute && !appearsToHaveObjectFiles(ruleContext.attributes())) {
        Artifact element = Iterables.getFirst(
            ccCompilationOutputs.getObjectFiles(false),
            ccCompilationOutputs.getObjectFiles(true).get(0));
        ruleContext.attributeWarning("srcs",
             "this library appears at first glance to have no object files, "
             + "but on closer inspection it does have something to link, e.g. "
             + element.prettyPrint() + ". "
             + "(You may have used some very confusing rule names in srcs? "
             + "Or the library consists entirely of a linker script?) "
             + "Bazel assumed linkstatic=1, but this may be inappropriate. "
             + "You may need to add an explicit '.cc' file to 'srcs'. "
             + "Alternatively, add 'linkstatic=1' to suppress this warning");
      }
    }
  }

  /**
   * Returns true if the rule (which must be a cc_library rule) appears to have object files.
   * This only looks at the rule itself, not at any other rules (from this package or other
   * packages) that it might reference.
   *
   * <p>In some cases, this may return "true" even though the rule actually has no object files.
   * For example, it will return true for a rule such as
   * <code>cc_library(name = 'foo', srcs = [':bar'])</code> because we can't tell what ':bar' is;
   * it might be a genrule that generates a source file, or it might be a genrule that generates a
   * header file. Likewise,
   * <code>cc_library(name = 'foo', srcs = select({':a': ['foo.cc'], ':b': []}))</code> returns
   * "true" even though the sources *may* be empty. This reflects the fact that there's no way
   * to tell which value "srcs" will take without knowing the rule's configuration.
   *
   * <p>In other cases, this may return "false" even though the rule actually does have object
   * files. For example, it will return false for a rule such as
   * <code>cc_library(name = 'foo', srcs = ['bar.h'])</code> but as in the other example above,
   * we can't tell whether 'bar.h' is a file name or a rule name, and 'bar.h' could in fact be the
   * name of a genrule that generates a source file.
   */
  public static boolean appearsToHaveObjectFiles(AttributeMap rule) {
    if ((rule instanceof RawAttributeMapper) && rule.isConfigurable("srcs")) {
      // Since this method gets called by loading phase logic (e.g. the cc_library implicit outputs
      // function), the attribute mapper may not be able to resolve configurable attributes. When
      // that's the case, there's no way to know which value a configurable "srcs" will take, so
      // we conservatively assume object files are possible.
      return true;
    }

    List<Label> srcs = rule.get("srcs", BuildType.LABEL_LIST);
    if (srcs != null) {
      for (Label srcfile : srcs) {
        /*
         * We cheat a little bit here by looking at the file extension
         * of the Label treated as file name.  In general that might
         * not necessarily work, because of the possibility that the
         * user might give a rule a funky name ending in one of these
         * extensions, e.g.
         *    genrule(name = 'foo.h', outs = ['foo.cc'], ...) // Funky rule name!
         *    cc_library(name = 'bar', srcs = ['foo.h']) // This DOES have object files.
         */
        if (!NO_OBJECT_GENERATING_FILETYPES.matches(srcfile.getName())) {
          return true;
        }
      }
    }
    return false;
  }
}
