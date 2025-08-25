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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A {@link TransitiveInfoProvider} so that {@code cc_toolchain} can pass {@link FdoSupport} to the
 * C++ rules.
 */
@Immutable
@AutoCodec
public class FdoSupportProvider implements TransitiveInfoProvider {
  private final FdoSupport fdoSupport;
  private final ProfileArtifacts profileArtifacts;
  private final ImmutableMap<PathFragment, Artifact> gcdaArtifacts;

  @AutoCodec.Instantiator
  public FdoSupportProvider(FdoSupport fdoSupport, ProfileArtifacts profileArtifacts,
      ImmutableMap<PathFragment, Artifact> gcdaArtifacts) {
    this.fdoSupport = fdoSupport;
    this.profileArtifacts = profileArtifacts;
    this.gcdaArtifacts = gcdaArtifacts;
  }

  public FdoSupport getFdoSupport() {
    return fdoSupport;
  }
  public Artifact getProfileArtifact() {
    return profileArtifacts != null ? profileArtifacts.getProfileArtifact() : null;
  }
  public Artifact getPrefetchHintsArtifact() {
    return profileArtifacts != null ? profileArtifacts.getPrefetchHintsArtifact() : null;
  }
  public ImmutableMap<PathFragment, Artifact> getGcdaArtifacts() {
    return gcdaArtifacts;
  }
}
