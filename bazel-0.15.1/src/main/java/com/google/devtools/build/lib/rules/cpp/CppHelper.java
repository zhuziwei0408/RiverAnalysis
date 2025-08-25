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

import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.NODEP_LABEL;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.MiddlemanFactory;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.analysis.AnalysisUtils;
import com.google.devtools.build.lib.analysis.Expander;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.PlatformConfiguration;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.StaticallyLinkedMarkerProvider;
import com.google.devtools.build.lib.analysis.ToolchainContext.ResolvedToolchainProviders;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.actions.SymlinkAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options.MakeVariableSource;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.RuleErrorConsumer;
import com.google.devtools.build.lib.rules.cpp.CcLinkParams.Linkstamp;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.FeatureConfiguration;
import com.google.devtools.build.lib.rules.cpp.CcToolchainFeatures.Tool;
import com.google.devtools.build.lib.rules.cpp.CppConfiguration.DynamicMode;
import com.google.devtools.build.lib.rules.cpp.Link.LinkTargetType;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Helper class for functionality shared by cpp related rules.
 *
 * <p>This class can be used only after the loading phase.
 */
public class CppHelper {

  static final PathFragment OBJS = PathFragment.create("_objs");
  static final PathFragment PIC_OBJS = PathFragment.create("_pic_objs");

  private static final String GREPPED_INCLUDES_SUFFIX = ".includes";

  // TODO(bazel-team): should this use Link.SHARED_LIBRARY_FILETYPES?
  public static final FileTypeSet SHARED_LIBRARY_FILETYPES = FileTypeSet.of(
      CppFileTypes.SHARED_LIBRARY,
      CppFileTypes.VERSIONED_SHARED_LIBRARY);

  private static final FileTypeSet CPP_FILETYPES = FileTypeSet.of(
      CppFileTypes.CPP_HEADER,
      CppFileTypes.CPP_SOURCE);

  private static final ImmutableList<String> LINKOPTS_PREREQUISITE_LABEL_KINDS =
      ImmutableList.of("deps", "srcs");

  /** Base label of the c++ toolchain category. */
  public static final String TOOLCHAIN_TYPE_LABEL = "//tools/cpp:toolchain_type";

  private CppHelper() {
    // prevents construction
  }

  /**
   * Returns the malloc implementation for the given target.
   */
  public static TransitiveInfoCollection mallocForTarget(
      RuleContext ruleContext, String mallocAttrName) {
    if (ruleContext.getFragment(CppConfiguration.class).customMalloc() != null) {
      return ruleContext.getPrerequisite(":default_malloc", Mode.TARGET);
    } else {
      return ruleContext.getPrerequisite(mallocAttrName, Mode.TARGET);
    }
  }

  public static TransitiveInfoCollection mallocForTarget(RuleContext ruleContext) {
    return mallocForTarget(ruleContext, "malloc");
  }

  /**
   * Returns true if this target should obtain c++ make variables from the toolchain instead of from
   * the configuration.
   */
  public static boolean shouldUseToolchainForMakeVariables(RuleContext ruleContext) {
    Label toolchainType = getToolchainTypeFromRuleClass(ruleContext);
    return (ruleContext.getConfiguration().getOptions().get(Options.class).makeVariableSource
            == MakeVariableSource.TOOLCHAIN)
        && (ruleContext
            .getFragment(PlatformConfiguration.class)
            .isToolchainTypeEnabled(toolchainType));
  }

  /**
   * Expands Make variables in a list of string and tokenizes the result. If the package feature
   * no_copts_tokenization is set, tokenize only items consisting of a single make variable.
   *
   * @param ruleContext the ruleContext to be used as the context of Make variable expansion
   * @param attributeName the name of the attribute to use in error reporting
   * @param input the list of strings to expand
   * @return a list of strings containing the expanded and tokenized values for the
   *         attribute
   */
  private static List<String> expandMakeVariables(
      RuleContext ruleContext, String attributeName, List<String> input) {
    boolean tokenization =
        !ruleContext.getFeatures().contains("no_copts_tokenization");

    List<String> tokens = new ArrayList<>();
    Expander expander = ruleContext.getExpander().withDataExecLocations();
    for (String token : input) {
      // Legacy behavior: tokenize all items.
      if (tokenization) {
        expander.tokenizeAndExpandMakeVars(tokens, attributeName, token);
      } else {
        String exp = expander.expandSingleMakeVariable(attributeName, token);
        if (exp != null) {
          try {
            ShellUtils.tokenize(tokens, exp);
          } catch (ShellUtils.TokenizationException e) {
            ruleContext.attributeError(attributeName, e.getMessage());
          }
        } else {
          tokens.add(expander.expand(attributeName, token));
        }
      }
    }
    return ImmutableList.copyOf(tokens);
  }

  /**
   * Returns the tokenized values of the copts attribute to copts.
   */
  // Called from CcCommon and CcSupport (Google's internal version of proto_library).
  public static ImmutableList<String> getAttributeCopts(RuleContext ruleContext) {
    String attr = "copts";
    Preconditions.checkArgument(ruleContext.getRule().isAttrDefined(attr, Type.STRING_LIST));
    List<String> unexpanded = ruleContext.attributes().get(attr, Type.STRING_LIST);
    return ImmutableList.copyOf(expandMakeVariables(ruleContext, attr, unexpanded));
  }

  // Called from CcCommon.
  static ImmutableList<String> getPackageCopts(RuleContext ruleContext) {
    List<String> unexpanded = ruleContext.getRule().getPackage().getDefaultCopts();
    return ImmutableList.copyOf(expandMakeVariables(ruleContext, "copts", unexpanded));
  }

  /**
   * Expands attribute value either using label expansion
   * (if attemptLabelExpansion == {@code true} and it does not look like make
   * variable or flag) or tokenizes and expands make variables.
   */
  public static List<String> expandLinkopts(
      RuleContext ruleContext, String attrName, Iterable<String> values) {
    List<String> result = new ArrayList<>();
    Expander expander = ruleContext.getExpander().withDataExecLocations();
    for (String value : values) {
      if (isLinkoptLabel(value)) {
        if (!expandLabel(ruleContext, result, value)) {
          ruleContext.attributeError(attrName, "could not resolve label '" + value + "'");
        }
      } else {
        expander
            .tokenizeAndExpandMakeVars(
                result,
                attrName,
                value);
      }
    }
    return result;
  }

  /**
   * Returns the immutable list of linker options for fully statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param config the CppConfiguration for this build
   * @param toolchain the c++ toolchain
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  public static ImmutableList<String> getFullyStaticLinkOptions(
      CppConfiguration config, CcToolchainProvider toolchain, boolean sharedLib) {
    if (sharedLib) {
      return toolchain.getSharedLibraryLinkOptions(
          toolchain.getLegacyMostlyStaticLinkFlags(
              config.getCompilationMode(), config.getLipoMode()));
    } else {
      return toolchain.getLegacyFullyStaticLinkFlags(
          config.getCompilationMode(), config.getLipoMode());
    }
  }

  /**
   * Returns the immutable list of linker options for mostly statically linked outputs. Does not
   * include command-line options passed via --linkopt or --linkopts.
   *
   * @param config the CppConfiguration for this build
   * @param toolchain the c++ toolchain
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  public static ImmutableList<String> getMostlyStaticLinkOptions(
      CppConfiguration config,
      CcToolchainProvider toolchain,
      boolean sharedLib,
      boolean shouldStaticallyLinkCppRuntimes) {
    if (sharedLib) {
      return toolchain.getSharedLibraryLinkOptions(
          shouldStaticallyLinkCppRuntimes
              ? toolchain.getLegacyMostlyStaticSharedLinkFlags(
                  config.getCompilationMode(), config.getLipoMode())
              : toolchain.getLegacyDynamicLinkFlags(
                  config.getCompilationMode(), config.getLipoMode()));
    } else {
      return toolchain.getLegacyMostlyStaticLinkFlags(
          config.getCompilationMode(), config.getLipoMode());
    }
  }

  /**
   * Returns the immutable list of linker options for artifacts that are not fully or mostly
   * statically linked. Does not include command-line options passed via --linkopt or --linkopts.
   *
   * @param config the CppConfiguration for this build
   * @param toolchain the c++ toolchain
   * @param sharedLib true if the output is a shared lib, false if it's an executable
   */
  public static ImmutableList<String> getDynamicLinkOptions(
      CppConfiguration config,
      CcToolchainProvider toolchain,
      Boolean sharedLib) {
    if (sharedLib) {
      return toolchain.getSharedLibraryLinkOptions(
          toolchain.getLegacyDynamicLinkFlags(config.getCompilationMode(), config.getLipoMode()));
    } else {
      return toolchain.getLegacyDynamicLinkFlags(config.getCompilationMode(), config.getLipoMode());
    }
  }

  /**
   * Determines if a linkopt can be a label. Linkopts come in 2 varieties:
   * literals -- flags like -Xl and makefile vars like $(LD) -- and labels,
   * which we should expand into filenames.
   *
   * @param linkopt the link option to test.
   * @return true if the linkopt is not a flag (starting with "-") or a makefile
   *         variable (starting with "$");
   */
  private static boolean isLinkoptLabel(String linkopt) {
    return !linkopt.startsWith("$") && !linkopt.startsWith("-");
  }

  /**
   * Expands a label against the target's deps, adding the expanded path strings
   * to the linkopts.
   *
   * @param linkopts the linkopts to add the expanded label to
   * @param labelName the name of the label to expand
   * @return true if the label was expanded successfully, false otherwise
   */
  private static boolean expandLabel(
      RuleContext ruleContext, List<String> linkopts, String labelName) {
    try {
      Label label = ruleContext.getLabel().getRelative(labelName);
      for (String prereqKind : LINKOPTS_PREREQUISITE_LABEL_KINDS) {
        for (TransitiveInfoCollection target : ruleContext
            .getPrerequisitesIf(prereqKind, Mode.TARGET, FileProvider.class)) {
          if (target.getLabel().equals(label)) {
            for (Artifact artifact : target.getProvider(FileProvider.class).getFilesToBuild()) {
              linkopts.add(artifact.getExecPathString());
            }
            return true;
          }
        }
      }
    } catch (LabelSyntaxException e) {
      // Quietly ignore and fall through.
    }
    linkopts.add(labelName);
    return false;
  }

  /**
   * Returns the dynamic linking mode (full, off, or default) implied by the given configuration and
   * toolchain.
   */
  public static DynamicMode getDynamicMode(CppConfiguration config, CcToolchainProvider toolchain) {
    // With LLVM, ThinLTO is automatically used in place of LIPO. ThinLTO works fine with dynamic
    // linking (and in fact creates a lot more work when dynamic linking is off).
    if (config.getLipoMode() == LipoMode.BINARY && !toolchain.isLLVMCompiler()) {
      // TODO(bazel-team): implement dynamic linking with LIPO
      return DynamicMode.OFF;
    } else {
      return config.getDynamicModeFlag();
    }
  }

  /** Returns true if LIPO optimization should be applied for this configuration and toolchain. */
  public static boolean isLipoOptimization(CppConfiguration config, CcToolchainProvider toolchain) {
    // The LIPO optimization bits are set in the LIPO context collector configuration, too.
    // If compiler is LLVM, then LIPO gets auto-converted to ThinLTO.
    return config.lipoOptimizationIsActivated() && !toolchain.isLLVMCompiler();
  }

  /**
   * Return {@link FdoSupportProvider} using default cc_toolchain attribute name.
   *
   * <p>Be careful to provide explicit attribute name if the rule doesn't store cc_toolchain under
   * the default name.
   */
  @Nullable
  public static FdoSupportProvider getFdoSupportUsingDefaultCcToolchainAttribute(
      RuleContext ruleContext) {
    return getFdoSupport(ruleContext, CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME);
  }

  @Nullable public static FdoSupportProvider getFdoSupport(RuleContext ruleContext,
      String ccToolchainAttribute) {
    return ruleContext
        .getPrerequisite(ccToolchainAttribute, Mode.TARGET)
        .getProvider(FdoSupportProvider.class);
  }

  public static NestedSet<Pair<String, String>> getCoverageEnvironmentIfNeeded(
      RuleContext ruleContext, CcToolchainProvider toolchain) {
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      return toolchain.getCoverageEnvironment();
    } else {
      return NestedSetBuilder.emptySet(Order.COMPILE_ORDER);
    }
  }

  public static NestedSet<Artifact> getGcovFilesIfNeeded(
      RuleContext ruleContext, CcToolchainProvider toolchain) {
    if (ruleContext.getConfiguration().isCodeCoverageEnabled()) {
      return toolchain.getCoverage();
    } else {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
  }

  /**
   * This almost trivial method looks up the default cc toolchain attribute on the rule context,
   * makes sure that it refers to a rule that has a {@link CcToolchainProvider} (gives an error
   * otherwise), and returns a reference to that {@link CcToolchainProvider}. The method only
   * returns {@code null} if there is no such attribute (this is currently not an error).
   *
   * <p>Be careful to provide explicit attribute name if the rule doesn't store cc_toolchain under
   * the default name.
   */
  @Nullable
  public static CcToolchainProvider getToolchainUsingDefaultCcToolchainAttribute(
      RuleContext ruleContext) {
    return getToolchain(ruleContext, CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME);
  }

  /**
   * Convenience function for finding the dynamic runtime inputs for the current toolchain. Useful
   * for non C++ rules that link against the C++ runtime.
   */
  public static NestedSet<Artifact> getDefaultCcToolchainDynamicRuntimeInputs(
      RuleContext ruleContext) {
    CcToolchainProvider defaultToolchain =
        getToolchain(ruleContext, CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME);
    if (defaultToolchain == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(ruleContext, defaultToolchain);
    return defaultToolchain.getDynamicRuntimeLinkInputs(featureConfiguration);
  }

  /**
   * Convenience function for finding the static runtime inputs for the current toolchain. Useful
   * for non C++ rules that link against the C++ runtime.
   */
  public static NestedSet<Artifact> getDefaultCcToolchainStaticRuntimeInputs(
      RuleContext ruleContext) {
    CcToolchainProvider defaultToolchain =
        getToolchain(ruleContext, CcToolchain.CC_TOOLCHAIN_DEFAULT_ATTRIBUTE_NAME);
    if (defaultToolchain == null) {
      return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    }
    FeatureConfiguration featureConfiguration =
        CcCommon.configureFeaturesOrReportRuleError(ruleContext, defaultToolchain);
    return defaultToolchain.getStaticRuntimeLinkInputs(featureConfiguration);
  }

  /**
   * Makes sure that the given info collection has a {@link CcToolchainProvider} (gives an error
   * otherwise), and returns a reference to that {@link CcToolchainProvider}. The method will only
   * return {@code null}, if the toolchain attribute is undefined for the rule class.
   */
  @Nullable
  public static CcToolchainProvider getToolchain(
      RuleContext ruleContext, String toolchainAttribute) {
     if (!ruleContext.isAttrDefined(toolchainAttribute, LABEL)) {
       // TODO(bazel-team): Report an error or throw an exception in this case.
       return null;
     }
     TransitiveInfoCollection dep = ruleContext.getPrerequisite(toolchainAttribute, Mode.TARGET);
     return getToolchain(ruleContext, dep);
   }

  /** Returns the c++ toolchain type, or null if it is not specified on the rule class. */
  public static Label getToolchainTypeFromRuleClass(RuleContext ruleContext) {
    Label toolchainType;
    // TODO(b/65835260): Remove this conditional once j2objc can learn the toolchain type.
    if (ruleContext.attributes().has(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME)) {
      toolchainType =
          ruleContext.attributes().get(CcToolchain.CC_TOOLCHAIN_TYPE_ATTRIBUTE_NAME, NODEP_LABEL);
    } else {
      toolchainType = null;
    }
    return toolchainType;
  }

  /**
   * Makes sure that the given info collection has a {@link CcToolchainProvider} (gives an error
   * otherwise), and returns a reference to that {@link CcToolchainProvider}. The method never
   * returns {@code null}, even if there is no toolchain.
   */
  public static CcToolchainProvider getToolchain(
      RuleContext ruleContext, TransitiveInfoCollection dep) {

    Label toolchainType = getToolchainTypeFromRuleClass(ruleContext);
    if (toolchainType != null
        && ruleContext
            .getFragment(PlatformConfiguration.class)
            .isToolchainTypeEnabled(toolchainType)) {
      return getToolchainFromPlatformConstraints(ruleContext, toolchainType);
    }
    return getToolchainFromCrosstoolTop(ruleContext, dep);
  }

  private static CcToolchainProvider getToolchainFromPlatformConstraints(
      RuleContext ruleContext, Label toolchainType) {
    ResolvedToolchainProviders providers =
        (ResolvedToolchainProviders)
            ruleContext.getToolchainContext().getResolvedToolchainProviders();
    return (CcToolchainProvider) providers.getForToolchainType(toolchainType);
  }

  private static CcToolchainProvider getToolchainFromCrosstoolTop(
      RuleContext ruleContext, TransitiveInfoCollection dep) {
    // TODO(bazel-team): Consider checking this generally at the attribute level.
    if ((dep == null) || (dep.get(ToolchainInfo.PROVIDER) == null)) {
      ruleContext.ruleError("The selected C++ toolchain is not a cc_toolchain rule");
      return CcToolchainProvider.EMPTY_TOOLCHAIN_IS_ERROR;
    }
    return (CcToolchainProvider) dep.get(ToolchainInfo.PROVIDER);
  }

  /** Returns the directory where object files are created. */
  public static PathFragment getObjDirectory(Label ruleLabel, boolean usePic) {
    if (usePic) {
      return AnalysisUtils.getUniqueDirectory(ruleLabel, PIC_OBJS);
    } else {
      return AnalysisUtils.getUniqueDirectory(ruleLabel, OBJS);
    }
  }

  /** Returns the directory where object files are created. */
  public static PathFragment getObjDirectory(Label ruleLabel) {
    return getObjDirectory(ruleLabel, false);
  }

  /**
   * A header that has been grepped for #include statements. Includes the original header as well as
   * a stripped version of that header that contains only #include statements.
   */
  @AutoCodec
  @AutoValue
  abstract static class PregreppedHeader {
    @AutoCodec.Instantiator
    static PregreppedHeader create(Artifact originalHeader, Artifact greppedHeader) {
      return new AutoValue_CppHelper_PregreppedHeader(originalHeader, greppedHeader);
    }

    /** Returns the original header, before grepping. */
    abstract Artifact originalHeader();

    /** Returns the grepped header, which contains only #include statements. */
    abstract Artifact greppedHeader();
  }

  /**
   * Creates a grep-includes ExtractInclusions action for generated sources/headers in the
   * needsIncludeScanning() BuildConfiguration case. Returns a map from original header Artifact to
   * the output Artifact of grepping over it. The return value only includes entries for generated
   * sources or headers when --extract_generated_inclusions is enabled.
   *
   * <p>Previously, incremental rebuilds redid all include scanning work for a given .cc source in
   * serial. For high-latency file systems, this could cause performance problems if many headers
   * are generated.
   */
  @Nullable
  public static final List<PregreppedHeader> createExtractInclusions(
      RuleContext ruleContext, CppSemantics semantics, Iterable<Artifact> prerequisites) {
    List<PregreppedHeader> extractions = new ArrayList<>();
    HashSet<Artifact> alreadyProcessedHeaders = new HashSet<>();
    for (Artifact prerequisite : prerequisites) {
      if (alreadyProcessedHeaders.contains(prerequisite)) {
        // Don't create duplicate actions just because user specified same header file twice.
        continue;
      }
      alreadyProcessedHeaders.add(prerequisite);

      Artifact scanned = createExtractInclusions(ruleContext, semantics, prerequisite);
      if (scanned != null) {
        extractions.add(PregreppedHeader.create(prerequisite, scanned));
      }
    }
    return extractions;
  }

  /**
   * Creates a grep-includes ExtractInclusions action for generated  sources/headers in the
   * needsIncludeScanning() BuildConfiguration case.
   *
   * <p>Previously, incremental rebuilds redid all include scanning work for a given
   * .cc source in serial. For high-latency file systems, this could cause
   * performance problems if many headers are generated.
   */
  private static final Artifact createExtractInclusions(
      RuleContext ruleContext, CppSemantics semantics, Artifact prerequisite) {
    if (ruleContext != null
        && semantics.needsIncludeScanning(ruleContext)
        && !prerequisite.isSourceArtifact()
        && CPP_FILETYPES.matches(prerequisite.getFilename())) {
      Artifact scanned = getIncludesOutput(ruleContext, prerequisite);
      Artifact grepIncludes = ruleContext.getPrerequisiteArtifact("$grep_includes", Mode.HOST);
      ruleContext.registerAction(
          new ExtractInclusionAction(
              ruleContext.getActionOwner(), prerequisite, scanned, grepIncludes));
      return scanned;
    }
    return null;
  }

  private static Artifact getIncludesOutput(RuleContext ruleContext, Artifact src) {
    Preconditions.checkArgument(!src.isSourceArtifact(), src);
    return ruleContext.getGenfilesArtifact(
        src.getRootRelativePath().replaceName(src.getFilename() + GREPPED_INCLUDES_SUFFIX));
  }

  /**
   * Returns the linked artifact.
   *
   * @param ruleContext the ruleContext to be used to scope the artifact
   * @param config the configuration to be used to scope the artifact
   * @param linkType the type of artifact, used to determine extension
   */
  public static Artifact getLinkedArtifact(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration config,
      LinkTargetType linkType)
      throws RuleErrorException {
    return getLinkedArtifact(
      ruleContext, ccToolchain, config, linkType, /* linkedArtifactNameSuffix= */ "");
  }

  /** Returns the linked artifact with the given suffix. */
  public static Artifact getLinkedArtifact(
      RuleContext ruleContext,
      CcToolchainProvider ccToolchain,
      BuildConfiguration config,
      LinkTargetType linkType,
      String linkedArtifactNameSuffix)
      throws RuleErrorException {
    PathFragment name = PathFragment.create(ruleContext.getLabel().getName());
    try {
      name =
          name.replaceName(
              getArtifactNameForCategory(
                  ruleContext,
                  ccToolchain,
                  linkType.getLinkerOutput(),
                  name.getBaseName()
                      + linkedArtifactNameSuffix
                      + linkType.getPicExtensionWhenApplicable()));
    } catch (RuleErrorException e) {
      ruleContext.throwWithRuleError("Cannot get linked artifact name: " + e.getMessage());
    }

    return ruleContext.getPackageRelativeArtifact(
        name, config.getBinDirectory(ruleContext.getRule().getRepository()));
  }

  public static Artifact getLinuxLinkedArtifact(
      RuleContext ruleContext,
      BuildConfiguration config,
      LinkTargetType linkType,
      String linkedArtifactNameSuffix) {
    PathFragment name = PathFragment.create(ruleContext.getLabel().getName());
    if (linkType != LinkTargetType.EXECUTABLE) {
      name = name.replaceName(
          "lib" + name.getBaseName()
              + linkedArtifactNameSuffix
              + linkType.getPicExtensionWhenApplicable()
              + linkType.getDefaultExtension());
    }

    return ruleContext.getPackageRelativeArtifact(
        name, config.getBinDirectory(ruleContext.getRule().getRepository()));
  }

  /**
   * Emits a warning on the rule if there are identical linkstamp artifacts with different {@code
   * CcCompilationContext}s.
   */
  public static void checkLinkstampsUnique(RuleErrorConsumer listener, CcLinkParams linkParams) {
    Map<Artifact, NestedSet<Artifact>> result = new LinkedHashMap<>();
    for (Linkstamp pair : linkParams.getLinkstamps()) {
      Artifact artifact = pair.getArtifact();
      if (result.containsKey(artifact)) {
        listener.ruleWarning("rule inherits the '" + artifact.toDetailString()
            + "' linkstamp file from more than one cc_library rule");
      }
      result.put(artifact, pair.getDeclaredIncludeSrcs());
    }
  }

  public static void addTransitiveLipoInfoForCommonAttributes(
      RuleContext ruleContext,
      CcCompilationOutputs outputs,
      NestedSetBuilder<IncludeScannable> scannableBuilder) {

    TransitiveLipoInfoProvider stl = null;
    if (ruleContext.getRule().getAttributeDefinition(":stl") != null
        && ruleContext.getPrerequisite(":stl", Mode.TARGET) != null) {
      // If the attribute is defined, it is never null.
      stl = ruleContext.getPrerequisite(":stl", Mode.TARGET)
          .getProvider(TransitiveLipoInfoProvider.class);
    }
    if (stl != null) {
      scannableBuilder.addTransitive(stl.getTransitiveIncludeScannables());
    }

    for (TransitiveLipoInfoProvider dep :
        ruleContext.getPrerequisites("deps", Mode.TARGET, TransitiveLipoInfoProvider.class)) {
      scannableBuilder.addTransitive(dep.getTransitiveIncludeScannables());
    }

    if (ruleContext.attributes().has("malloc", LABEL)) {
      TransitiveInfoCollection malloc = mallocForTarget(ruleContext);
      TransitiveLipoInfoProvider provider = malloc.getProvider(TransitiveLipoInfoProvider.class);
      if (provider != null) {
        scannableBuilder.addTransitive(provider.getTransitiveIncludeScannables());
      }
    }

    for (IncludeScannable scannable : outputs.getLipoScannables()) {
      Preconditions.checkState(scannable.getIncludeScannerSources().size() == 1);
      scannableBuilder.add(scannable);
    }
  }

  // TODO(bazel-team): figure out a way to merge these 2 methods. See the Todo in
  // CcCommonConfiguredTarget.noCoptsMatches().
  /**
   * Determines if we should apply -fPIC for this rule's C++ compilations. This determination is
   * generally made by the global C++ configuration settings "needsPic" and "usePicForBinaries".
   * However, an individual rule may override these settings by applying -fPIC" to its "nocopts"
   * attribute. This allows incompatible rules to "opt out" of global PIC settings (see bug:
   * "Provide a way to turn off -fPIC for targets that can't be built that way").
   *
   * @param ruleContext the context of the rule to check
   * @return true if this rule's compilations should apply -fPIC, false otherwise
   */
  public static boolean usePicForDynamicLibraries(
      RuleContext ruleContext, CcToolchainProvider toolchain) {
    return ruleContext.getFragment(CppConfiguration.class).forcePic()
        || toolchain.toolchainNeedsPic();
  }

  /** Returns whether binaries must be compiled with position independent code. */
  public static boolean usePicForBinaries(RuleContext ruleContext, CcToolchainProvider toolchain) {
    CppConfiguration config = ruleContext.getFragment(CppConfiguration.class);
    if (CcCommon.noCoptsMatches("-fPIC", ruleContext)) {
      return false;
    }
    return config.forcePic()
        || (toolchain.toolchainNeedsPic() && config.getCompilationMode() != CompilationMode.OPT);
  }

  /**
   * Returns true if shared libraries must be compiled with position independent code for the build
   * implied by the given config and toolchain.
   */

  /**
   * Returns the LIPO context provider for configured target,
   * or null if such a provider doesn't exist.
   */
  public static LipoContextProvider getLipoContextProvider(RuleContext ruleContext) {
    if (ruleContext
            .getRule()
            .getAttributeDefinition(TransitiveLipoInfoProvider.LIPO_CONTEXT_COLLECTOR)
        == null) {
      return null;
    }

    TransitiveInfoCollection dep =
        ruleContext.getPrerequisite(
            TransitiveLipoInfoProvider.LIPO_CONTEXT_COLLECTOR, Mode.DONT_CHECK);
    return (dep != null) ? dep.getProvider(LipoContextProvider.class) : null;
  }

  /**
   * Creates a CppModuleMap object for pure c++ builds. The module map artifact becomes a candidate
   * input to a CppCompileAction.
   */
  public static CppModuleMap createDefaultCppModuleMap(RuleContext ruleContext, String suffix) {
    // Create the module map artifact as a genfile.
    Artifact mapFile =
        ruleContext.getPackageRelativeArtifact(
            ruleContext.getLabel().getName()
                + suffix
                + Iterables.getOnlyElement(CppFileTypes.CPP_MODULE_MAP.getExtensions()),
            ruleContext
                .getConfiguration()
                .getGenfilesDirectory(ruleContext.getRule().getRepository()));
    return new CppModuleMap(mapFile, ruleContext.getLabel().toString());
  }

  /**
   * Returns a middleman for all files to build for the given configured target, substituting shared
   * library artifacts with corresponding solib symlinks. If multiple calls are made, then it
   * returns the same artifact for configurations with the same internal directory.
   *
   * <p>The resulting middleman only aggregates the inputs and must be expanded before populating
   * the set of files necessary to execute an action.
   */
  static List<Artifact> getAggregatingMiddlemanForCppRuntimes(
      RuleContext ruleContext,
      String purpose,
      Iterable<Artifact> artifacts,
      String solibDir,
      String solibDirOverride,
      BuildConfiguration configuration) {
    return getMiddlemanInternal(
        ruleContext,
        ruleContext.getActionOwner(),
        purpose,
        artifacts,
        true,
        true,
        solibDir,
        solibDirOverride,
        configuration);
  }

  @VisibleForTesting
  public static List<Artifact> getAggregatingMiddlemanForTesting(
      RuleContext ruleContext,
      ActionOwner owner,
      String purpose,
      Iterable<Artifact> artifacts,
      boolean useSolibSymlinks,
      String solibDir,
      BuildConfiguration configuration) {
    return getMiddlemanInternal(
        ruleContext,
        owner,
        purpose,
        artifacts,
        useSolibSymlinks,
        false,
        solibDir,
        null,
        configuration);
  }

  /** Internal implementation for getAggregatingMiddlemanForCppRuntimes. */
  private static List<Artifact> getMiddlemanInternal(
      RuleContext ruleContext,
      ActionOwner actionOwner,
      String purpose,
      Iterable<Artifact> artifacts,
      boolean useSolibSymlinks,
      boolean isCppRuntime,
      String solibDir,
      String solibDirOverride,
      BuildConfiguration configuration) {
    MiddlemanFactory factory = ruleContext.getAnalysisEnvironment().getMiddlemanFactory();
    if (useSolibSymlinks) {
      List<Artifact> symlinkedArtifacts = new ArrayList<>();
      for (Artifact artifact : artifacts) {
        Preconditions.checkState(Link.SHARED_LIBRARY_FILETYPES.matches(artifact.getFilename()));
        symlinkedArtifacts.add(
            isCppRuntime
                ? SolibSymlinkAction.getCppRuntimeSymlink(
                    ruleContext, artifact, solibDir, solibDirOverride, configuration)
                : SolibSymlinkAction.getDynamicLibrarySymlink(
                    ruleContext, solibDir, artifact, false, true, configuration));
      }
      artifacts = symlinkedArtifacts;
      purpose += "_with_solib";
    }
    return ImmutableList.of(
        factory.createMiddlemanAllowMultiple(ruleContext.getAnalysisEnvironment(), actionOwner,
            ruleContext.getPackageDirectory(), purpose, artifacts,
            configuration.getMiddlemanDirectory(ruleContext.getRule().getRepository())));
  }

  /**
   * Returns the FDO build subtype.
   */
  public static String getFdoBuildStamp(RuleContext ruleContext, FdoSupport fdoSupport) {
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    if (fdoSupport.isAutoFdoEnabled()) {
      return (cppConfiguration.getLipoMode() == LipoMode.BINARY) ? "ALIPO" : "AFDO";
    }
    if (cppConfiguration.isFdo()) {
      return (cppConfiguration.getLipoMode() == LipoMode.BINARY) ? "LIPO" : "FDO";
    }
    return null;
  }

  /**
   * Returns a relative path to the bin directory for data in AutoFDO LIPO mode.
   */
  public static PathFragment getLipoDataBinFragment(BuildConfiguration configuration) {
    PathFragment parent = configuration.getBinFragment().getParentDirectory();
    return parent.replaceName(parent.getBaseName() + "-lipodata")
        .getChild(configuration.getBinFragment().getBaseName());
  }

  /**
   * Returns a relative path to the genfiles directory for data in AutoFDO LIPO mode.
   */
  public static PathFragment getLipoDataGenfilesFragment(BuildConfiguration configuration) {
    PathFragment parent = configuration.getGenfilesFragment().getParentDirectory();
    return parent.replaceName(parent.getBaseName() + "-lipodata")
        .getChild(configuration.getGenfilesFragment().getBaseName());
  }

  /** Creates an action to strip an executable. */
  public static void createStripAction(
      RuleContext context,
      CcToolchainProvider toolchain,
      CppConfiguration cppConfiguration,
      Artifact input,
      Artifact output,
      FeatureConfiguration featureConfiguration) {
    if (featureConfiguration.isEnabled(CppRuleClasses.NO_STRIPPING)) {
      context.registerAction(
          new SymlinkAction(
              context.getActionOwner(),
              input,
              output,
              "Symlinking original binary as stripped binary"));
      return;
    }

    if (!featureConfiguration.actionIsConfigured(CppCompileAction.STRIP_ACTION_NAME)) {
      context.ruleError("Expected action_config for 'strip' to be configured.");
      return;
    }

    Tool stripTool =
        Preconditions.checkNotNull(
            featureConfiguration.getToolForAction(CppCompileAction.STRIP_ACTION_NAME));
    CcToolchainVariables variables =
        new CcToolchainVariables.Builder(toolchain.getBuildVariables())
            .addStringVariable(
                StripBuildVariables.OUTPUT_FILE.getVariableName(), output.getExecPathString())
            .addStringSequenceVariable(
                StripBuildVariables.STRIPOPTS.getVariableName(), cppConfiguration.getStripOpts())
            .addStringVariable(CcCommon.INPUT_FILE_VARIABLE_NAME, input.getExecPathString())
            .build();
    ImmutableList<String> commandLine =
        ImmutableList.copyOf(
            featureConfiguration.getCommandLine(CppCompileAction.STRIP_ACTION_NAME, variables));
    ImmutableMap.Builder<String, String> executionInfoBuilder = ImmutableMap.builder();
    for (String executionRequirement : stripTool.getExecutionRequirements()) {
      executionInfoBuilder.put(executionRequirement, "");
    }
    Action[] stripAction =
        new SpawnAction.Builder()
            .addInput(input)
            .addTransitiveInputs(toolchain.getStrip())
            .addOutput(output)
            .useDefaultShellEnvironment()
            .setExecutable(stripTool.getToolPathFragment())
            .setExecutionInfo(executionInfoBuilder.build())
            .setProgressMessage("Stripping %s for %s", output.prettyPrint(), context.getLabel())
            .setMnemonic("CcStrip")
            .addCommandLine(CustomCommandLine.builder().addAll(commandLine).build())
            .build(context);
    context.registerAction(stripAction);
  }

  public static void maybeAddStaticLinkMarkerProvider(RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext) {
    boolean staticallyLinked = false;
    CppConfiguration cppConfiguration = ruleContext.getFragment(CppConfiguration.class);
    if (ruleContext.getFeatures().contains("fully_static_link")) {
      staticallyLinked = true;
    } else if (cppConfiguration.hasStaticLinkOption()) {
      staticallyLinked = true;
    } else if (ruleContext.attributes().has("linkopts", Type.STRING_LIST)
        && ruleContext.attributes().get("linkopts", Type.STRING_LIST).contains("-static")) {
      staticallyLinked = true;
    }

    if (staticallyLinked) {
      builder.add(StaticallyLinkedMarkerProvider.class, new StaticallyLinkedMarkerProvider(true));
    }
  }

  static Artifact getCompileOutputArtifact(RuleContext ruleContext, String outputName,
      BuildConfiguration config) {
    PathFragment objectDir = getObjDirectory(ruleContext.getLabel());
    return ruleContext.getDerivedArtifact(objectDir.getRelative(outputName),
        config.getBinDirectory(ruleContext.getRule().getRepository()));
  }

  /** Returns the corresponding compiled TreeArtifact given the source TreeArtifact. */
  public static SpecialArtifact getCompileOutputTreeArtifact(
      RuleContext ruleContext, Artifact sourceTreeArtifact, String outputName, boolean usePic) {
    PathFragment objectDir = getObjDirectory(ruleContext.getLabel(), usePic);

    return ruleContext.getTreeArtifact(
        objectDir.getRelative(outputName), sourceTreeArtifact.getRoot());
  }

  public static String getArtifactNameForCategory(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      ArtifactCategory category,
      String outputName)
      throws RuleErrorException {
    try {
      return toolchain.getFeatures().getArtifactNameForCategory(category, outputName);
    } catch (InvalidConfigurationException e) {
      ruleContext.throwWithRuleError(e.getMessage());
      throw new IllegalStateException("Should not be reached");
    }
  }

  static String getDotdFileName(
      RuleContext ruleContext,
      CcToolchainProvider toolchain,
      ArtifactCategory outputCategory,
      String outputName)
      throws RuleErrorException {
    String baseName = outputCategory == ArtifactCategory.OBJECT_FILE
        || outputCategory == ArtifactCategory.PROCESSED_HEADER
        ? outputName
        : getArtifactNameForCategory(ruleContext, toolchain, outputCategory, outputName);

    return getArtifactNameForCategory(
        ruleContext, toolchain, ArtifactCategory.INCLUDED_FILE_LIST, baseName);
  }

  /**
   * Returns true when {@link CppRuleClasses#WINDOWS_EXPORT_ALL_SYMBOLS} feature is enabled and
   * {@link CppRuleClasses#NO_WINDOWS_EXPORT_ALL_SYMBOLS} feature is not enabled and no custom DEF
   * file is specified in win_def_file attribute.
   */
  public static boolean shouldUseGeneratedDefFile(
      RuleContext ruleContext, FeatureConfiguration featureConfiguration) {
    return featureConfiguration.isEnabled(CppRuleClasses.WINDOWS_EXPORT_ALL_SYMBOLS)
        && !featureConfiguration.isEnabled(CppRuleClasses.NO_WINDOWS_EXPORT_ALL_SYMBOLS)
        && ruleContext.getPrerequisiteArtifact("win_def_file", Mode.TARGET) == null;
  }

  /**
   * Create actions for parsing object files to generate a DEF file, should only be used when
   * targeting Windows.
   *
   * @param defParser The tool we use to parse object files for generating the DEF file.
   * @param objectFiles A list of object files to parse
   * @param dllName The DLL name to be written into the DEF file, it specifies which DLL is required
   *     at runtime
   * @return The DEF file artifact.
   */
  public static Artifact createDefFileActions(
      RuleContext ruleContext,
      Artifact defParser,
      ImmutableList<Artifact> objectFiles,
      String dllName) {
    Artifact defFile = ruleContext.getBinArtifact(ruleContext.getLabel().getName() + ".def");
    CustomCommandLine.Builder argv = new CustomCommandLine.Builder();
    for (Artifact objectFile : objectFiles) {
      argv.addDynamicString(objectFile.getExecPathString());
    }

    ruleContext.registerAction(
        new SpawnAction.Builder()
            .addInputs(objectFiles)
            .addOutput(defFile)
            .setExecutable(defParser)
            .useDefaultShellEnvironment()
            .addCommandLine(
                CustomCommandLine.builder().addExecPath(defFile).addDynamicString(dllName).build())
            .addCommandLine(
                argv.build(),
                ParamFileInfo.builder(ParameterFile.ParameterFileType.SHELL_QUOTED)
                    .setCharset(UTF_8)
                    .setUseAlways(true)
                    .build())
            .setMnemonic("DefParser")
            .build(ruleContext));
    return defFile;
  }

  /**
   * Returns true if the build implied by the given config and toolchain uses --start-lib/--end-lib
   * ld options.
   */
  public static boolean useStartEndLib(CppConfiguration config, CcToolchainProvider toolchain) {
    return config.startEndLibIsRequested() && toolchain.supportsStartEndLib();
  }

  /**
   * Returns the type of archives being used by the build implied by the given config and toolchain.
   */
  public static Link.ArchiveType getArchiveType(
      CppConfiguration config, CcToolchainProvider toolchain) {
    return useStartEndLib(config, toolchain)
        ? Link.ArchiveType.START_END_LIB
        : Link.ArchiveType.REGULAR;
  }

  /**
   * Returns true if interface shared objects should be used in the build implied by the given
   * config and toolchain.
   */
  public static boolean useInterfaceSharedObjects(
      CppConfiguration config, CcToolchainProvider toolchain) {
    return toolchain.supportsInterfaceSharedObjects() && config.getUseInterfaceSharedObjects();
  }

}
