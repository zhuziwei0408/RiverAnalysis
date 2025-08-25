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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;

/**
 * Stub action to be used as the generating action for FDO files that are extracted from the
 * FDO zip.
 *
 * <p>This is needed because the extraction is currently not a bona fide action, therefore, Blaze
 * would complain that these files have no generating action if we did not set it to an instance of
 * this class.
 *
 * <p>These actions are all owned by the {@code cc_toolchain} rule. It's possible that they are
 * shared actions since the FDO path is the same regardless of the configuration of the rule that
 * created it, but in practice this shouldn't happen very often, because we usually have only one
 * FDO optimized configuration.
 */
@Immutable
public final class FdoStubAction extends AbstractAction {
  public FdoStubAction(ActionOwner owner, Artifact output) {
    // TODO(bazel-team): Make extracting the zip file a honest-to-God action so that we can do away
    // with this ugliness.
    super(owner, ImmutableList.<Artifact>of(), ImmutableList.of(output));
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext) {
    return ActionResult.EMPTY;
  }

  @Override
  public String getMnemonic() {
    return "FdoStubAction";
  }

  @Override
  protected void computeKey(ActionKeyContext actionKeyContext, Fingerprint fp) {
    fp.addString("fdoStubAction");
  }

  @Override
  public void prepare(FileSystem fileSystem, Path execRoot) {
    // The superclass would delete the output files here. We can't let that happen, since this
    // action does not in fact create those files; it is only a placeholder and the actual files
    // are created *before* the execution phase in FdoSupport.extractFdoZip()
  }
}
