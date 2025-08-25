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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.cache.Metadata;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.skyframe.FileArtifactValue;
import com.google.devtools.build.lib.skyframe.FileContentsProxy;
import com.google.devtools.build.lib.vfs.FileStatus;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Tree;
import java.io.IOException;

/** A fake implementation of the {@link ActionInputFileCache} interface. */
final class FakeActionInputFileCache implements ActionInputFileCache {
  private final Path execRoot;
  private final BiMap<ActionInput, String> cas = HashBiMap.create();
  private final DigestUtil digestUtil;

  FakeActionInputFileCache(Path execRoot) {
    this.execRoot = execRoot;
    this.digestUtil = new DigestUtil(execRoot.getFileSystem().getDigestFunction());
  }

  @Override
  public Metadata getMetadata(ActionInput input) throws IOException {
    String hexDigest = Preconditions.checkNotNull(cas.get(input), input);
    Path path = execRoot.getRelative(input.getExecPath());
    FileStatus stat = path.stat(Symlinks.FOLLOW);
    return FileArtifactValue.createNormalFile(
        HashCode.fromString(hexDigest).asBytes(), FileContentsProxy.create(stat), stat.getSize());
  }

  void setDigest(ActionInput input, String digest) {
    cas.put(input, digest);
  }

  public Digest createScratchInput(ActionInput input, String content) throws IOException {
    Path inputFile = execRoot.getRelative(input.getExecPath());
    FileSystemUtils.createDirectoryAndParents(inputFile.getParentDirectory());
    FileSystemUtils.writeContentAsLatin1(inputFile, content);
    Digest digest = digestUtil.compute(inputFile);
    setDigest(input, digest.getHash());
    return digest;
  }

  public Digest createScratchInputDirectory(ActionInput input, Tree content) throws IOException {
    Path inputFile = execRoot.getRelative(input.getExecPath());
    FileSystemUtils.createDirectoryAndParents(inputFile);
    Digest digest = digestUtil.compute(content);
    setDigest(input, digest.getHash());
    return digest;
  }
}
