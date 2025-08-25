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

import com.google.devtools.build.lib.skylarkbuildapi.ProviderApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

/**
 * Utilies related to C++ support.
 */
@SkylarkModule(name = "cc_common", doc = "Utilities related to C++ support.")
public interface CcModuleApi {

  @SkylarkCallable(
    name = "CcToolchainInfo",
    doc =
        "The key used to retrieve the provider that contains information about the C++ "
            + "toolchain being usCced",
    structField = true
  )
  public ProviderApi getCcToolchainProvider();

  @Deprecated
  @SkylarkCallable(
      name = "do_not_use_tools_cpp_compiler_present",
      doc =
          "Do not use this field, its only puprose is to help with migration from "
              + "config_setting.values{'compiler') to "
              + "config_settings.flag_values{'@bazel_tools//tools/cpp:compiler'}",
      structField = true)
  default void compilerFlagExists() {}
}
