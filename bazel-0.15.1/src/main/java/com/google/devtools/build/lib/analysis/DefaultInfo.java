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
package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/** DefaultInfo is provided by all targets implicitly and contains all standard fields. */
@Immutable
public final class DefaultInfo extends NativeInfo {

  // Accessors for Skylark
  private static final String DATA_RUNFILES_FIELD = "data_runfiles";
  private static final String DEFAULT_RUNFILES_FIELD = "default_runfiles";
  private static final String FILES_FIELD = "files";
  private static final ImmutableList<String> KEYS =
      ImmutableList.of(
          DATA_RUNFILES_FIELD,
          DEFAULT_RUNFILES_FIELD,
          FILES_FIELD,
          FilesToRunProvider.SKYLARK_NAME);

  private final RunfilesProvider runfilesProvider;
  private final FileProvider fileProvider;
  private final FilesToRunProvider filesToRunProvider;
  private final AtomicReference<SkylarkNestedSet> files = new AtomicReference<>();

  public static final String SKYLARK_NAME = "DefaultInfo";

  // todo(dslomov,vladmos): make this provider return DefaultInfo.
  public static final NativeProvider<NativeInfo> PROVIDER =
      new NativeProvider<NativeInfo>(NativeInfo.class, SKYLARK_NAME) {
        @Override
        protected NativeInfo createInstanceFromSkylark(
            Object[] args, Environment env, Location loc) {
          @SuppressWarnings("unchecked")
          Map<String, Object> kwargs = (Map<String, Object>) args[0];
          return new NativeInfo(this, kwargs, loc);
        }
      };

  private DefaultInfo(
      RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    super(PROVIDER);
    this.runfilesProvider = runfilesProvider;
    this.fileProvider = fileProvider;
    this.filesToRunProvider = filesToRunProvider;
  }

  public static DefaultInfo build(
      RunfilesProvider runfilesProvider,
      FileProvider fileProvider,
      FilesToRunProvider filesToRunProvider) {
    return new DefaultInfo(runfilesProvider, fileProvider, filesToRunProvider);
  }

  @Override
  public Object getValue(String name) {
    switch (name) {
      case DATA_RUNFILES_FIELD:
        return (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDataRunfiles();
      case DEFAULT_RUNFILES_FIELD:
        return (runfilesProvider == null) ? Runfiles.EMPTY : runfilesProvider.getDefaultRunfiles();
      case FILES_FIELD:
        if (files.get() == null) {
          files.compareAndSet(
              null, SkylarkNestedSet.of(Artifact.class, fileProvider.getFilesToBuild()));
        }
        return files.get();
      case FilesToRunProvider.SKYLARK_NAME:
        return filesToRunProvider;
    }
    return null;
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return KEYS;
  }
}
