// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Strings;
import com.google.common.collect.FluentIterable;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ParamFileInfo;
import com.google.devtools.build.lib.actions.ParameterFile.ParameterFileType;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

/** Builder for the action that generates the R class for libraries. */
public class LibraryRGeneratorActionBuilder {
  private String javaPackage;
  private Iterable<ValidatedAndroidData> deps = ImmutableList.of();
  private ResourceContainer resourceContainer;
  private Artifact rJavaClassJar;

  public LibraryRGeneratorActionBuilder setJavaPackage(String javaPackage) {
    this.javaPackage = javaPackage;
    return this;
  }

  public LibraryRGeneratorActionBuilder withPrimary(ResourceContainer resourceContainer) {
    this.resourceContainer = resourceContainer;
    return this;
  }

  public LibraryRGeneratorActionBuilder withDependencies(ResourceDependencies resourceDeps) {
    this.deps = resourceDeps.getResourceContainers();
    return this;
  }

  public LibraryRGeneratorActionBuilder setClassJarOut(Artifact rJavaClassJar) {
    this.rJavaClassJar = rJavaClassJar;
    return this;
  }

  public ResourceContainer build(RuleContext ruleContext) {
    AndroidSdkProvider sdk = AndroidSdkProvider.fromRuleContext(ruleContext);

    CustomCommandLine.Builder builder = new CustomCommandLine.Builder();
    NestedSetBuilder<Artifact> inputs = NestedSetBuilder.naiveLinkOrder();
    FilesToRunProvider executable =
        ruleContext.getExecutablePrerequisite("$android_resources_busybox", Mode.HOST);
    inputs.addAll(executable.getRunfilesSupport().getRunfilesArtifacts());

    builder.add("--tool").add("GENERATE_LIBRARY_R").add("--");

    if (!Strings.isNullOrEmpty(javaPackage)) {
      builder.add("--packageForR", javaPackage);
    }

    FluentIterable<ValidatedAndroidData> symbolProviders =
        FluentIterable.from(deps).append(resourceContainer);

    if (!symbolProviders.isEmpty()) {
      ImmutableList<Artifact> symbols =
          symbolProviders.stream().map(ValidatedAndroidData::getSymbols).collect(toImmutableList());
      builder.addExecPaths("--symbols", symbols);
      inputs.addTransitive(NestedSetBuilder.wrap(Order.NAIVE_LINK_ORDER, symbols));
    }

    builder.addExecPath("--classJarOutput", rJavaClassJar);
    builder.addLabel("--targetLabel", ruleContext.getLabel());

    builder.addExecPath("--androidJar", sdk.getAndroidJar());
    inputs.add(sdk.getAndroidJar());

    // Create the spawn action.
    SpawnAction.Builder spawnActionBuilder = new SpawnAction.Builder();
    ruleContext.registerAction(
        spawnActionBuilder
            .useDefaultShellEnvironment()
            .addTransitiveInputs(inputs.build())
            .addOutputs(ImmutableList.<Artifact>of(rJavaClassJar))
            .addCommandLine(
                builder.build(), ParamFileInfo.builder(ParameterFileType.SHELL_QUOTED).build())
            .setExecutable(executable)
            .setProgressMessage("Generating Library R Classes: %s", ruleContext.getLabel())
            .setMnemonic("LibraryRClassGenerator")
            .build(ruleContext));
    return resourceContainer.toBuilder().setJavaClassJar(rJavaClassJar).build();
  }
}
