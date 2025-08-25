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

import com.google.common.base.Function;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcRunfilesApi;

/**
 * Runfiles for C++ targets.
 *
 * <p>Contains two {@link Runfiles} objects: one for the statically linked binary and one for the
 * dynamically linked binary. Both contain dynamic libraries needed at runtime and data
 * dependencies.
 */
@Immutable
@AutoCodec
public final class CcRunfiles implements CcRunfilesApi {

  private final Runfiles runfilesForLinkingStatically;
  private final Runfiles runfilesForLinkingDynamically;

  @AutoCodec.Instantiator
  public CcRunfiles(Runfiles runfilesForLinkingStatically, Runfiles runfilesForLinkingDynamically) {
    this.runfilesForLinkingStatically = runfilesForLinkingStatically;
    this.runfilesForLinkingDynamically = runfilesForLinkingDynamically;
  }

  @Override
  public Runfiles getRunfilesForLinkingStatically() {
    return runfilesForLinkingStatically;
  }

  @Override
  public Runfiles getRunfilesForLinkingDynamically() {
    return runfilesForLinkingDynamically;
  }

  /**
   * Returns a function that gets the static C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> RUNFILES_FOR_LINKING_STATICALLY =
      input -> {
        CcLinkingInfo provider = input.get(CcLinkingInfo.PROVIDER);
        CcRunfiles ccRunfiles = provider == null ? null : provider.getCcRunfiles();
        return ccRunfiles == null ? Runfiles.EMPTY : ccRunfiles.getRunfilesForLinkingStatically();
      };

  /**
   * Returns a function that gets the shared C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles>
      RUNFILES_FOR_LINKING_DYNAMICALLY =
          input -> {
            CcLinkingInfo provider = input.get(CcLinkingInfo.PROVIDER);
            CcRunfiles ccRunfiles = provider == null ? null : provider.getCcRunfiles();
            return ccRunfiles == null
                ? Runfiles.EMPTY
                : ccRunfiles.getRunfilesForLinkingDynamically();
          };

  /**
   * Returns a function that gets the C++ runfiles from a {@link TransitiveInfoCollection} or
   * the empty runfiles instance if it does not contain that provider.
   */
  public static final Function<TransitiveInfoCollection, Runfiles> runfilesFunction(
      boolean linkingStatically) {
    return linkingStatically ? RUNFILES_FOR_LINKING_STATICALLY : RUNFILES_FOR_LINKING_DYNAMICALLY;
  }
}
