// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainInfoApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;

/**
 * Information about the C++ toolchain.
 */
@SkylarkModule(name = "CcToolchainInfo", doc = "Information about the C++ compiler being used.")
public interface CcToolchainProviderApi<PathFragmentT> extends ToolchainInfoApi {

  @SkylarkCallable(
      name = "built_in_include_directories",
      doc = "Returns the list of built-in directories of the compiler.",
      structField = true
  )
  public ImmutableList<PathFragmentT> getBuiltInIncludeDirectories();

  @SkylarkCallable(
    name = "sysroot",
    structField = true,
    doc =
        "Returns the sysroot to be used. If the toolchain compiler does not support "
            + "different sysroots, or the sysroot is the same as the default sysroot, then "
            + "this method returns <code>None</code>."
  )
  public PathFragmentT getSysroot();

  @SkylarkCallable(name = "compiler", structField = true, doc = "C++ compiler.",
      allowReturnNones = true)
  public String getCompiler();

  @SkylarkCallable(name = "libc", structField = true, doc = "libc version string.",
      allowReturnNones = true)
  public String getTargetLibc();

  @SkylarkCallable(name = "cpu", structField = true, doc = "Target CPU of the C++ toolchain.",
      allowReturnNones = true)
  public String getTargetCpu();

  @SkylarkCallable(
      name = "unfiltered_compiler_options",
      doc =
          "Returns the default list of options which cannot be filtered by BUILD "
              + "rules. These should be appended to the command line after filtering.")
  // TODO(b/24373706): Remove this method once new C++ toolchain API is available
  public ImmutableList<String> getUnfilteredCompilerOptionsWithSysroot(
      Iterable<String> featuresNotUsedAnymore);

  @SkylarkCallable(
    name = "link_options_do_not_use",
    structField = true,
    doc =
        "Returns the set of command-line linker options, including any flags "
            + "inferred from the command-line options."
  )
  public ImmutableList<String> getLinkOptionsWithSysroot();

  @SkylarkCallable(
    name = "target_gnu_system_name",
    structField = true,
    doc = "The GNU System Name.",
    allowReturnNones = true
  )
  public String getTargetGnuSystemName();

  @SkylarkCallable(
      name = "compiler_options",
      doc =
          "Returns the default options to use for compiling C, C++, and assembler. "
              + "This is just the options that should be used for all three languages. "
              + "There may be additional C-specific or C++-specific options that should be used, "
              + "in addition to the ones returned by this method"
  )
  public ImmutableList<String> getCompilerOptions();

  @SkylarkCallable(
      name = "c_options",
      doc =
          "Returns the list of additional C-specific options to use for compiling C. "
              + "These should be go on the command line after the common options returned by "
              + "<code>compiler_options</code>")
  public ImmutableList<String> getCOptions();

  @SkylarkCallable(
      name = "cxx_options",
      doc =
          "Returns the list of additional C++-specific options to use for compiling C++. "
              + "These should be go on the command line after the common options returned by "
              + "<code>compiler_options</code>")
  @Deprecated
  public ImmutableList<String> getCxxOptionsWithCopts();

  @SkylarkCallable(
      name = "fully_static_link_options",
      doc =
          "Returns the immutable list of linker options for fully statically linked "
              + "outputs. Does not include command-line options passed via --linkopt or "
              + "--linkopts.")
  @Deprecated
  public ImmutableList<String> getFullyStaticLinkOptions(Boolean sharedLib) throws EvalException;

  @SkylarkCallable(
      name = "mostly_static_link_options",
      doc =
          "Returns the immutable list of linker options for mostly statically linked "
              + "outputs. Does not include command-line options passed via --linkopt or "
              + "--linkopts.")
  @Deprecated
  public ImmutableList<String> getMostlyStaticLinkOptions(Boolean sharedLib);

  @SkylarkCallable(
      name = "dynamic_link_options",
      doc =
          "Returns the immutable list of linker options for artifacts that are not "
              + "fully or mostly statically linked. Does not include command-line options "
              + "passed via --linkopt or --linkopts."
  )
  @Deprecated
  public ImmutableList<String> getDynamicLinkOptions(Boolean sharedLib);
}
