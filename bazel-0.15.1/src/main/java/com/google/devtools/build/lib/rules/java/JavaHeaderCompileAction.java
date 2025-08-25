// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLines;
import com.google.devtools.build.lib.actions.CommandLines.CommandLineLimits;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.ParameterFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Action for Java header compilation, to be used if --java_header_compilation is enabled.
 *
 * <p>The header compiler consumes the inputs of a java compilation, and produces an interface jar
 * that can be used as a compile-time jar by upstream targets. The header interface jar is
 * equivalent to the output of ijar, but unlike ijar the header compiler operates directly on Java
 * source files instead post-processing the class outputs of the compilation. Compiling the
 * interface jar from source moves javac off the build's critical path.
 *
 * <p>The implementation of the header compiler tool can be found under {@code
 * //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine}.
 */
@AutoCodec
public class JavaHeaderCompileAction extends SpawnAction {

  private static final String GUID = "952db158-2654-4ced-87e5-4646d50523cf";

  private static final ResourceSet LOCAL_RESOURCES =
      ResourceSet.createWithRamCpuIo(/*memoryMb=*/ 750.0, /*cpuUsage=*/ 0.5, /*ioUsage=*/ 0.0);

  private final Iterable<Artifact> directInputs;
  @Nullable private final CommandLine directCommandLine;

  /** The command line for a direct classpath compilation, or {@code null} if disabled. */
  @VisibleForTesting
  @Nullable
  public CommandLine directCommandLine() {
    return directCommandLine;
  }

  /**
   * Constructs an action to compile a set of Java source files to a header interface jar.
   *
   * @param owner the action owner, typically a java_* RuleConfiguredTarget
   * @param tools the set of files comprising the tool that creates the header interface jar
   * @param directInputs the set of direct input artifacts of the compile action
   * @param inputs the set of transitive input artifacts of the compile action
   * @param outputs the outputs of the action
   * @param primaryOutput the output jar
   * @param commandLines the transitive command line arguments for the java header compiler
   * @param directCommandLine the direct command line arguments for the java header compiler
   * @param commandLineLimits the command line limits
   * @param progressMessage the message printed during the progression of the build
   */
  protected JavaHeaderCompileAction(
      ActionOwner owner,
      Iterable<Artifact> tools,
      Iterable<Artifact> directInputs,
      Iterable<Artifact> inputs,
      Iterable<Artifact> outputs,
      Artifact primaryOutput,
      CommandLines commandLines,
      CommandLine directCommandLine,
      CommandLineLimits commandLineLimits,
      CharSequence progressMessage,
      RunfilesSupplier runfilesSupplier) {
    super(
        owner,
        tools,
        inputs,
        outputs,
        primaryOutput,
        LOCAL_RESOURCES,
        commandLines,
        commandLineLimits,
        false,
        // TODO(#3320): This is missing the config's action environment.
        JavaCompileAction.UTF8_ACTION_ENVIRONMENT,
        /* executionInfo= */ ImmutableMap.of(),
        progressMessage,
        runfilesSupplier,
        "Turbine",
        /* executeUnconditionally= */ false,
        /* extraActionInfoSupplier= */ null);
    this.directInputs = checkNotNull(directInputs);
    this.directCommandLine = checkNotNull(directCommandLine);
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString(GUID);
    try {
      super.computeKey(actionKeyContext, fp);
      fp.addStrings(directCommandLine.arguments());
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("JavaHeaderCompileAction command line expansion cannot fail");
    }
  }

  @Override
  protected List<SpawnResult> internalExecute(ActionExecutionContext actionExecutionContext)
      throws ExecException, InterruptedException {
    Spawn spawn = getDirectSpawn();
    SpawnActionContext context = actionExecutionContext.getContext(SpawnActionContext.class);
    try {
      return context.exec(spawn, actionExecutionContext);
    } catch (ExecException e) {
      // if the direct input spawn failed, try again with transitive inputs to produce better
      // better messages
      try {
        return context.exec(getSpawn(actionExecutionContext), actionExecutionContext);
      } catch (CommandLineExpansionException commandLineExpansionException) {
        throw new UserExecException(commandLineExpansionException);
      }
      // The compilation should never fail with direct deps but succeed with transitive inputs
      // unless it failed due to a strict deps error, in which case fall back to the transitive
      // classpath may allow it to succeed (Strict Java Deps errors are reported by javac,
      // not turbine).
    }
  }

  private final Spawn getDirectSpawn() {
    try {
      return new BaseSpawn(
          ImmutableList.copyOf(directCommandLine.arguments()),
          ImmutableMap.<String, String>of() /*environment*/,
          ImmutableMap.<String, String>of() /*executionInfo*/,
          this,
          LOCAL_RESOURCES) {
        @Override
        public Iterable<? extends ActionInput> getInputFiles() {
          return directInputs;
        }
      };
    } catch (CommandLineExpansionException e) {
      throw new AssertionError("JavaHeaderCompileAction command line expansion cannot fail");
    }
  }

  /** Builder class to construct Java header compilation actions. */
  public static class Builder {

    private final RuleContext ruleContext;

    private Artifact outputJar;
    @Nullable private Artifact outputDepsProto;
    private ImmutableSet<Artifact> sourceFiles = ImmutableSet.of();
    private final Collection<Artifact> sourceJars = new ArrayList<>();
    private NestedSet<Artifact> classpathEntries =
        NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private ImmutableList<Artifact> bootclasspathEntries = ImmutableList.<Artifact>of();
    @Nullable private Label targetLabel;
    @Nullable private String injectingRuleKind;
    private PathFragment tempDirectory;
    private BuildConfiguration.StrictDepsMode strictJavaDeps
        = BuildConfiguration.StrictDepsMode.OFF;
    private NestedSet<Artifact> directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
    private NestedSet<Artifact> compileTimeDependencyArtifacts =
        NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    private ImmutableList<String> javacOpts;
    private NestedSet<Artifact> processorPath = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    private final List<String> processorNames = new ArrayList<>();

    private NestedSet<Artifact> additionalInputs = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
    private Artifact javacJar;
    private NestedSet<Artifact> toolsJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);

    public Builder(RuleContext ruleContext) {
      this.ruleContext = ruleContext;
    }

    /** Sets the output jdeps file. */
    public Builder setOutputDepsProto(@Nullable Artifact outputDepsProto) {
      this.outputDepsProto = outputDepsProto;
      return this;
    }

    /** Sets the direct dependency artifacts. */
    public Builder setDirectJars(NestedSet<Artifact> directJars) {
      checkNotNull(directJars, "directJars must not be null");
      this.directJars = directJars;
      return this;
    }

    /** Sets the .jdeps artifacts for direct dependencies. */
    public Builder setCompileTimeDependencyArtifacts(NestedSet<Artifact> dependencyArtifacts) {
      checkNotNull(dependencyArtifacts, "dependencyArtifacts must not be null");
      this.compileTimeDependencyArtifacts = dependencyArtifacts;
      return this;
    }

    /** Sets Java compiler flags. */
    public Builder setJavacOpts(ImmutableList<String> javacOpts) {
      checkNotNull(javacOpts, "javacOpts must not be null");
      this.javacOpts = javacOpts;
      return this;
    }

    /** Sets the output jar. */
    public Builder setOutputJar(Artifact outputJar) {
      checkNotNull(outputJar, "outputJar must not be null");
      this.outputJar = outputJar;
      return this;
    }

    /** Adds Java source files to compile. */
    public Builder setSourceFiles(ImmutableSet<Artifact> sourceFiles) {
      checkNotNull(sourceFiles, "sourceFiles must not be null");
      this.sourceFiles = sourceFiles;
      return this;
    }

    /** Adds a jar archive of Java sources to compile. */
    public Builder addSourceJars(Collection<Artifact> sourceJars) {
      checkNotNull(sourceJars, "sourceJars must not be null");
      this.sourceJars.addAll(sourceJars);
      return this;
    }

    /** Sets the compilation classpath entries. */
    public Builder setClasspathEntries(NestedSet<Artifact> classpathEntries) {
      checkNotNull(classpathEntries, "classpathEntries must not be null");
      this.classpathEntries = classpathEntries;
      return this;
    }

    /** Sets the compilation bootclasspath entries. */
    public Builder setBootclasspathEntries(ImmutableList<Artifact> bootclasspathEntries) {
      checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
      this.bootclasspathEntries = bootclasspathEntries;
      return this;
    }

    /** Sets the annotation processors classpath entries. */
    public Builder setProcessorPaths(NestedSet<Artifact> processorPaths) {
      checkNotNull(processorPaths, "processorPaths must not be null");
      this.processorPath = processorPaths;
      return this;
    }

    /** Sets the fully-qualified class names of annotation processors to run. */
    public Builder addProcessorNames(Collection<String> processorNames) {
      checkNotNull(processorNames, "processorNames must not be null");
      this.processorNames.addAll(processorNames);
      return this;
    }

    /** Sets the label of the target being compiled. */
    public Builder setTargetLabel(@Nullable Label targetLabel) {
      this.targetLabel = targetLabel;
      return this;
    }

    /** Sets the injecting rule kind of the target being compiled. */
    public Builder setInjectingRuleKind(@Nullable String injectingRuleKind) {
      this.injectingRuleKind = injectingRuleKind;
      return this;
    }

    /**
     * Sets the path to a temporary directory, e.g. for extracting sourcejar entries to before
     * compilation.
     */
    public Builder setTempDirectory(PathFragment tempDirectory) {
      checkNotNull(tempDirectory, "tempDirectory must not be null");
      this.tempDirectory = tempDirectory;
      return this;
    }

    /** Sets the Strict Java Deps mode. */
    public Builder setStrictJavaDeps(BuildConfiguration.StrictDepsMode strictJavaDeps) {
      checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
      this.strictJavaDeps = strictJavaDeps;
      return this;
    }

    /** Sets the javabase inputs. */
    public Builder setAdditionalInputs(NestedSet<Artifact> additionalInputs) {
      checkNotNull(additionalInputs, "additionalInputs must not be null");
      this.additionalInputs = additionalInputs;
      return this;
    }

    /** Sets the javac jar. */
    public Builder setJavacJar(Artifact javacJar) {
      checkNotNull(javacJar, "javacJar must not be null");
      this.javacJar = javacJar;
      return this;
    }

    /** Sets the tools jars. */
    public Builder setToolsJars(NestedSet<Artifact> toolsJars) {
      checkNotNull(toolsJars, "toolsJars must not be null");
      this.toolsJars = toolsJars;
      return this;
    }

    /** Builds and registers the {@link JavaHeaderCompileAction} for a header compilation. */
    public void build(JavaToolchainProvider javaToolchain, JavaRuntimeInfo hostJavabase) {
      checkNotNull(outputDepsProto, "outputDepsProto must not be null");
      checkNotNull(sourceFiles, "sourceFiles must not be null");
      checkNotNull(sourceJars, "sourceJars must not be null");
      checkNotNull(classpathEntries, "classpathEntries must not be null");
      checkNotNull(bootclasspathEntries, "bootclasspathEntries must not be null");
      checkNotNull(tempDirectory, "tempDirectory must not be null");
      checkNotNull(strictJavaDeps, "strictJavaDeps must not be null");
      checkNotNull(directJars, "directJars must not be null");
      checkNotNull(
          compileTimeDependencyArtifacts, "compileTimeDependencyArtifacts must not be null");
      checkNotNull(javacOpts, "javacOpts must not be null");
      checkNotNull(processorPath, "processorPath must not be null");
      checkNotNull(processorNames, "processorNames must not be null");

      // Invariant: if strictJavaDeps is OFF, then directJars and
      // dependencyArtifacts are ignored
      if (strictJavaDeps == BuildConfiguration.StrictDepsMode.OFF) {
        directJars = NestedSetBuilder.emptySet(Order.NAIVE_LINK_ORDER);
        compileTimeDependencyArtifacts = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      }

      // The compilation uses API-generating annotation processors and has to fall back to
      // javac-turbine.
      boolean requiresAnnotationProcessing = !processorNames.isEmpty();

      NestedSet<Artifact> tools =
          NestedSetBuilder.<Artifact>stableOrder()
              .add(javacJar)
              .addTransitive(javaToolchain.getHeaderCompiler().getFilesToRun())
              .addTransitive(toolsJars)
              .build();
      ImmutableList<Artifact> outputs = ImmutableList.of(outputJar, outputDepsProto);
      NestedSet<Artifact> baseInputs =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(hostJavabase.javaBaseInputsMiddleman())
              .addTransitive(additionalInputs)
              .addAll(bootclasspathEntries)
              .addAll(sourceJars)
              .addAll(sourceFiles)
              .addTransitive(tools)
              .build();

      boolean noFallback =
          ruleContext.getFragment(JavaConfiguration.class).headerCompilationDisableJavacFallback();
      // The action doesn't require annotation processing and either javac-turbine fallback is
      // disabled, or the action doesn't distinguish between direct and transitive deps, so
      // use a plain SpawnAction to invoke turbine.
      if ((noFallback || directJars.isEmpty()) && !requiresAnnotationProcessing) {
        SpawnAction.Builder builder = new SpawnAction.Builder();
        NestedSet<Artifact> classpath;
        final ParamFileInfo paramFileInfo;
        if (!directJars.isEmpty() || classpathEntries.isEmpty()) {
          classpath = directJars;
          paramFileInfo = null;
        } else {
          classpath = classpathEntries;
          // Transitive classpath actions may exceed the command line length limit.
          paramFileInfo =
              ParamFileInfo.builder(ParameterFileType.UNQUOTED).setUseAlways(true).build();
        }
        CustomCommandLine.Builder commandLine =
            baseCommandLine(CustomCommandLine.builder(), classpath);
        if (noFallback) {
          commandLine.add("--nojavac_fallback");
        }
        Artifact headerCompiler = javaToolchain.getHeaderCompiler().getExecutable();
        // The header compiler is either a jar file that needs to be executed using
        // `java -jar <path>`, or an executable that can be run directly.
        if (!headerCompiler.getExtension().equals("jar")) {
          builder.setExecutable(headerCompiler);
          builder.addTool(javaToolchain.getHeaderCompiler());
        } else {
          builder.setJarExecutable(
              hostJavabase.javaBinaryExecPath(), headerCompiler, javaToolchain.getJvmOptions());
        }
        ruleContext.registerAction(
            builder
                .addTransitiveTools(tools)
                .addTransitiveInputs(baseInputs)
                .addTransitiveInputs(classpath)
                .addOutputs(outputs)
                .addCommandLine(commandLine.build(), paramFileInfo)
                .setMnemonic("Turbine")
                .setProgressMessage(getProgressMessage())
                .build(ruleContext));
        return;
      }

      CustomCommandLine.Builder commandLine = getBaseArgs(javaToolchain, hostJavabase);
      CommandLine paramFileCommandLine = transitiveCommandLine();
      NestedSetBuilder<Artifact> transitiveInputs =
          NestedSetBuilder.<Artifact>stableOrder()
              .addTransitive(baseInputs)
              .addTransitive(classpathEntries)
              .addTransitive(processorPath)
              .addTransitive(compileTimeDependencyArtifacts);
      final CommandLines commandLines;

      if (ruleContext.getConfiguration().deferParamFiles()) {
        commandLines =
            CommandLines.builder()
                .addCommandLine(commandLine.build())
                .addCommandLine(
                    paramFileCommandLine,
                    ParamFileInfo.builder(ParameterFile.ParameterFileType.UNQUOTED)
                        .setUseAlways(true)
                        .setCharset(ISO_8859_1)
                        .build())
                .build();
      } else {
        PathFragment paramFilePath = ParameterFile.derivePath(outputJar.getRootRelativePath());
        Artifact paramsFile =
            ruleContext
                .getAnalysisEnvironment()
                .getDerivedArtifact(paramFilePath, outputJar.getRoot());
        transitiveInputs.add(paramsFile);
        commandLine.addFormatted("@%s", paramsFile.getExecPath());
        commandLines = CommandLines.of(commandLine.build());
        ParameterFileWriteAction parameterFileWriteAction =
            new ParameterFileWriteAction(
                ruleContext.getActionOwner(),
                paramsFile,
                paramFileCommandLine,
                ParameterFile.ParameterFileType.UNQUOTED,
                ISO_8859_1);
        ruleContext.registerAction(parameterFileWriteAction);
      }

      if (requiresAnnotationProcessing) {
        // turbine doesn't support API-generating annotation processors, so skip the two-tiered
        // turbine/javac-turbine action and just use SpawnAction to invoke javac-turbine.
        ruleContext.registerAction(
            new SpawnAction(
                ruleContext.getActionOwner(),
                tools,
                transitiveInputs.build(),
                outputs,
                outputJar,
                LOCAL_RESOURCES,
                commandLines,
                ruleContext.getConfiguration().getCommandLineLimits(),
                false,
                // TODO(b/63280599): This is missing the config's action environment.
                JavaCompileAction.UTF8_ACTION_ENVIRONMENT,
                /* executionInfo= */ ImmutableMap.of(),
                getProgressMessageWithAnnotationProcessors(),
                javaToolchain.getHeaderCompiler().getRunfilesSupplier(),
                "JavacTurbine",
                /* executeUnconditionally= */ false,
                /* extraActionInfoSupplier= */ null));
        return;
      }

      // The action doesn't require annotation processing, javac-turbine fallback is enabled, and
      // the target distinguishes between direct and transitive deps. Try a two-tiered spawn
      // the invokes turbine with direct deps, and falls back to javac-turbine on failures to
      // produce better diagnostics. (At the cost of slower failed actions and a larger
      // cache footprint.)
      // TODO(cushon): productionize --nojavac_fallback and remove this path
      checkState(!directJars.isEmpty());
      NestedSet<Artifact> directInputs =
          NestedSetBuilder.fromNestedSet(baseInputs).addTransitive(directJars).build();
      CustomCommandLine directCommandLine = baseCommandLine(
          getBaseArgs(javaToolchain, hostJavabase), directJars)
          .build();
      ruleContext.registerAction(
          new JavaHeaderCompileAction(
              ruleContext.getActionOwner(),
              tools,
              directInputs,
              transitiveInputs.build(),
              outputs,
              outputJar,
              commandLines,
              directCommandLine,
              ruleContext.getConfiguration().getCommandLineLimits(),
              getProgressMessage(),
              javaToolchain.getHeaderCompiler().getRunfilesSupplier()));
    }

    private LazyString getProgressMessageWithAnnotationProcessors() {
      List<String> shortNames = new ArrayList<>();
      for (String name : processorNames) {
        shortNames.add(name.substring(name.lastIndexOf('.') + 1));
      }
      String tail = " and running annotation processors (" + Joiner.on(", ").join(shortNames) + ")";
      return getProgressMessage(tail);
    }

    private LazyString getProgressMessage() {
      return getProgressMessage("");
    }

    private LazyString getProgressMessage(String tail) {
      Artifact outputJar = this.outputJar;
      int fileCount = sourceFiles.size() + sourceJars.size();
      return new LazyString() {
        @Override
        public String toString() {
          return String.format(
              "Compiling Java headers %s (%d files)%s", outputJar.prettyPrint(), fileCount, tail);
        }
      };
    }

    private CustomCommandLine.Builder getBaseArgs(
        JavaToolchainProvider javaToolchain, JavaRuntimeInfo hostJavabase) {
      Artifact headerCompiler = javaToolchain.getHeaderCompiler().getExecutable();
      if (!headerCompiler.getExtension().equals("jar")) {
        return CustomCommandLine.builder().addExecPath(headerCompiler);
      } else {
        return CustomCommandLine.builder()
            .addPath(hostJavabase.javaBinaryExecPath())
            .add("-Xverify:none")
            .addAll(javaToolchain.getJvmOptions())
            .addExecPath("-jar", headerCompiler);
      }
    }

    /**
     * Adds the command line arguments shared by direct classpath and transitive classpath
     * invocations.
     */
    private CustomCommandLine.Builder baseCommandLine(
        CustomCommandLine.Builder result, NestedSet<Artifact> classpathEntries) {
      result.addExecPath("--output", outputJar);

      if (outputDepsProto != null) {
        result.addExecPath("--output_deps", outputDepsProto);
      }

      result.add("--temp_dir").addPath(tempDirectory);

      result.addExecPaths("--bootclasspath", bootclasspathEntries);

      result.addExecPaths("--sources", sourceFiles);

      if (!sourceJars.isEmpty()) {
        result.addExecPaths("--source_jars", ImmutableList.copyOf(sourceJars));
      }

      if (!javacOpts.isEmpty()) {
        result.addAll("--javacopts", javacOpts);
        // terminate --javacopts with `--` to support javac flags that start with `--`
        result.add("--");
      }

      if (targetLabel != null) {
        result.add("--target_label");
        if (targetLabel.getPackageIdentifier().getRepository().isDefault()
            || targetLabel.getPackageIdentifier().getRepository().isMain()) {
          result.addLabel(targetLabel);
        } else {
          // @-prefixed strings will be assumed to be params filenames and expanded,
          // so add an extra @ to escape it.
          result.addPrefixedLabel("@", targetLabel);
        }
      }
      if (injectingRuleKind != null) {
        result.add("--injecting_rule_kind", injectingRuleKind);
      }
      result.addExecPaths("--classpath", classpathEntries);
      return result;
    }

    /** Builds a transitive classpath command line. */
    private CommandLine transitiveCommandLine() {
      CustomCommandLine.Builder result = CustomCommandLine.builder();
      baseCommandLine(result, classpathEntries);
      if (!processorNames.isEmpty()) {
        result.addAll("--processors", ImmutableList.copyOf(processorNames));
      }
      if (!processorPath.isEmpty()) {
        result.addExecPaths("--processorpath", processorPath);
      }
      if (strictJavaDeps != BuildConfiguration.StrictDepsMode.OFF) {
        result.addExecPaths("--direct_dependencies", directJars);
        if (!compileTimeDependencyArtifacts.isEmpty()) {
          result.addExecPaths("--deps_artifacts", compileTimeDependencyArtifacts);
        }
      }
      return result.build();
    }
  }
}
