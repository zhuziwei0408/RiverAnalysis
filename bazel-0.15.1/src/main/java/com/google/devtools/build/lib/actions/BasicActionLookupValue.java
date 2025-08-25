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
package com.google.devtools.build.lib.actions;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.common.base.MoreObjects.ToStringHelper;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import java.util.ArrayList;

/**
 * Basic implementation of {@link ActionLookupValue} where the value itself owns and maintains
 * the list of generating actions.
 */
public class BasicActionLookupValue extends ActionLookupValue {
  protected final ArrayList<ActionAnalysisMetadata> actions;
  @VisibleForSerialization protected final ImmutableMap<Artifact, Integer> generatingActionIndex;

  protected BasicActionLookupValue(
      ActionAnalysisMetadata action,
      boolean removeActionAfterEvaluation) {
    this(Actions.GeneratingActions.fromSingleAction(action), removeActionAfterEvaluation);
  }

  protected BasicActionLookupValue(
      ArrayList<ActionAnalysisMetadata> actions,
      ImmutableMap<Artifact, Integer> generatingActionIndex,
      boolean removeActionsAfterEvaluation) {
    super(removeActionsAfterEvaluation);
    this.actions = actions;
    this.generatingActionIndex = generatingActionIndex;
  }

  @VisibleForTesting
  public BasicActionLookupValue(
      Actions.GeneratingActions generatingActions, boolean removeActionsAfterEvaluation) {
    this(new ArrayList<>(generatingActions.getActions()),
        generatingActions.getGeneratingActionIndex(),
        removeActionsAfterEvaluation);
  }

  @Override
  protected ArrayList<ActionAnalysisMetadata> getActions() {
    return actions;
  }

  @Override
  protected ImmutableMap<Artifact, Integer> getGeneratingActionIndex() {
    return generatingActionIndex;
  }

  protected ToStringHelper getStringHelper() {
    return MoreObjects.toStringHelper(this)
        .add("actions", actions)
        .add("generatingActionIndex", generatingActionIndex);
  }
}
