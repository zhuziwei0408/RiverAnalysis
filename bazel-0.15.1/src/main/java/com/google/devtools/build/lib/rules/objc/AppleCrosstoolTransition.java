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

package com.google.devtools.build.lib.rules.objc;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.rules.apple.AppleCommandLineOptions;
import com.google.devtools.build.lib.rules.apple.AppleConfiguration;
import com.google.devtools.build.lib.rules.apple.ApplePlatform;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/**
 * Transition that produces a configuration that causes c++ toolchain selection to use the
 * CROSSTOOL given in apple_crosstool_top.
 */
public class AppleCrosstoolTransition implements PatchTransition {

  /** A singleton instance of AppleCrosstoolTransition. */
  @AutoCodec
  public static final PatchTransition APPLE_CROSSTOOL_TRANSITION = new AppleCrosstoolTransition();

  @Override
  public BuildOptions patch(BuildOptions buildOptions) {
    BuildOptions result = buildOptions.clone();

    if (!appleCrosstoolTransitionIsAppliedForAllObjc(buildOptions)) {
      return buildOptions;
    }

    AppleCommandLineOptions appleOptions = buildOptions.get(AppleCommandLineOptions.class);
    BuildConfiguration.Options configOptions = buildOptions.get(BuildConfiguration.Options.class);

    String cpu =
        ApplePlatform.cpuStringForTarget(
            appleOptions.applePlatformType,
            determineSingleArchitecture(appleOptions, configOptions));
    setAppleCrosstoolTransitionConfiguration(buildOptions, result, cpu);
    return result;
  }
  
  /**
   * Sets configuration fields required for a transition that uses apple_crosstool_top in place of
   * the default CROSSTOOL.
   *
   * @param from options from the originating configuration
   * @param to options for the destination configuration. This instance will be modified
   *     to so the destination configuration uses the apple crosstool
   * @param cpu {@code --cpu} value for toolchain selection in the destination configuration
   */
  public static void setAppleCrosstoolTransitionConfiguration(BuildOptions from,
      BuildOptions to, String cpu) {
    to.get(BuildConfiguration.Options.class).cpu = cpu;
    to.get(CppOptions.class).crosstoolTop =
        from.get(AppleCommandLineOptions.class).appleCrosstoolTop;
    to.get(AppleCommandLineOptions.class).targetUsesAppleCrosstool = true;

    // --compiler = "compiler" for all OSX toolchains.  We do not support asan/tsan, cfi, etc. on
    // darwin.
    to.get(CppOptions.class).cppCompiler = "compiler";

    // OSX toolchains always use the runtime of the platform they are targeting (i.e. we do not
    // support custom production environments).
    to.get(CppOptions.class).libcTopLabel = null;
    to.get(CppOptions.class).glibc = null;

    // OSX toolchains do not support fission.
    to.get(CppOptions.class).fissionModes = ImmutableList.of();
  }

  /**
   * Returns true if the given options imply use of AppleCrosstoolTransition for all apple targets.
   */
  public static boolean appleCrosstoolTransitionIsAppliedForAllObjc(BuildOptions options) {
    return options.get(AppleCommandLineOptions.class).enableAppleCrosstoolTransition;
  }

  /**
   * Returns the Apple architecture implied by AppleCommandLineOptions and
   * BuildConfiguration.Options
   */
  private String determineSingleArchitecture(
      AppleCommandLineOptions appleOptions, BuildConfiguration.Options configOptions) {
    if (!Strings.isNullOrEmpty(appleOptions.appleSplitCpu)) {
      return appleOptions.appleSplitCpu;
    }
    switch (appleOptions.applePlatformType) {
      case IOS:
        if (!appleOptions.iosMultiCpus.isEmpty()) {
          return appleOptions.iosMultiCpus.get(0);
        } else {
          return AppleConfiguration.iosCpuFromCpu(configOptions.cpu);
        }
      case WATCHOS:
        return appleOptions.watchosCpus.get(0);
      case TVOS:
        return appleOptions.tvosCpus.get(0);
      case MACOS:
        return appleOptions.macosCpus.get(0);
      default:
        throw new IllegalArgumentException(
            "Unhandled platform type " + appleOptions.applePlatformType);
    }
  }
}
