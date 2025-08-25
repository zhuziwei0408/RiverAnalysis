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

import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/** Interface for Runfiles of C++ targets. */
@SkylarkModule(
    name = "cc_runfiles",
    category = SkylarkModuleCategory.NONE,
    doc =
        "CC runfiles. Separated into files needed for linking statically and files needed for "
            + "linking dynamically.")
public interface CcRunfilesApi {
  @SkylarkCallable(name = "runfiles_for_linking_statically", documented = false, structField = true)
  Runfiles getRunfilesForLinkingStatically();

  @SkylarkCallable(
      name = "runfiles_for_linking_dynamically",
      documented = false,
      structField = true)
  Runfiles getRunfilesForLinkingDynamically();
}
