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
package com.google.devtools.build.lib.rules.android;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nullable;

/**
 * Interface used to indicate a container for resources, assets, or both can be merged.
 *
 * <p>Currently, resources and assets invoke the same action for merging, so virtually all the code
 * to generate that action is identical. Implementing this interface allows it to be reused.
 */
public interface MergableAndroidData {

  /** @return the roots of all resources to be merged */
  default ImmutableList<PathFragment> getResourceRoots() {
    return ImmutableList.of();
  }

  /** @return the roots of all assets to be merged */
  default ImmutableList<PathFragment> getAssetRoots() {
    return ImmutableList.of();
  }

  default ImmutableList<Artifact> getResources() {
    return ImmutableList.of();
  }

  default ImmutableList<Artifact> getAssets() {
    return ImmutableList.of();
  }

  Label getLabel();

  Artifact getSymbols();

  @Nullable
  Artifact getCompiledSymbols();
}
