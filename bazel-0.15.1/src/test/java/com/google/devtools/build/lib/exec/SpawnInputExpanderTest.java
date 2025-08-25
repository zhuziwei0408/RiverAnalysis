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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.exec.FilesetManifest.RelativeSymlinkBehavior.ERROR;
import static com.google.devtools.build.lib.exec.FilesetManifest.RelativeSymlinkBehavior.IGNORE;
import static com.google.devtools.build.lib.exec.FilesetManifest.RelativeSymlinkBehavior.RESOLVE;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.EmptyRunfilesSupplier;
import com.google.devtools.build.lib.actions.FilesetOutputSymlink;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesSupplierImpl;
import com.google.devtools.build.lib.exec.util.FakeActionInputFileCache;
import com.google.devtools.build.lib.skyframe.FileArtifactValue;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link SpawnInputExpander}.
 */
@RunWith(JUnit4.class)
public class SpawnInputExpanderTest {
  private static final byte[] FAKE_DIGEST = new byte[] {1, 2, 3, 4};

  private FileSystem fs;
  private Path execRoot;
  private SpawnInputExpander expander;
  private Map<PathFragment, ActionInput> inputMappings;

  @Before
  public final void createSpawnInputExpander() throws Exception  {
    fs = new InMemoryFileSystem();
    execRoot = fs.getPath("/root");
    expander = new SpawnInputExpander(execRoot, /*strict=*/ true);
    inputMappings = Maps.newHashMap();
  }

  private void scratchFile(String file, String... lines) throws Exception {
    Path path = fs.getPath(file);
    FileSystemUtils.createDirectoryAndParents(path.getParentDirectory());
    FileSystemUtils.writeLinesAs(path, StandardCharsets.UTF_8, lines);
  }

  @Test
  public void testEmptyRunfiles() throws Exception {
    RunfilesSupplier supplier = EmptyRunfilesSupplier.INSTANCE;
    expander.addRunfilesToInputs(inputMappings, supplier, null);
    assertThat(inputMappings).isEmpty();
  }

  @Test
  public void testRunfilesSingleFile() throws Exception {
    Artifact artifact =
        new Artifact(
            fs.getPath("/root/dir/file"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    FakeActionInputFileCache mockCache = new FakeActionInputFileCache();
    mockCache.put(artifact, FileArtifactValue.createNormalFile(FAKE_DIGEST, 0));

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/file"), artifact);
  }

  @Test
  public void testRunfilesDirectoryStrict() throws Exception {
    Artifact artifact =
        new Artifact(
            fs.getPath("/root/dir/file"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    FakeActionInputFileCache mockCache = new FakeActionInputFileCache();
    mockCache.put(artifact, FileArtifactValue.createDirectory(-1));

    try {
      expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
      fail();
    } catch (IOException expected) {
      assertThat(expected.getMessage().contains("Not a file: /root/dir/file")).isTrue();
    }
  }

  @Test
  public void testRunfilesDirectoryNonStrict() throws Exception {
    Artifact artifact =
        new Artifact(
            fs.getPath("/root/dir/file"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Runfiles runfiles = new Runfiles.Builder("workspace").addArtifact(artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    FakeActionInputFileCache mockCache = new FakeActionInputFileCache();
    mockCache.put(artifact, FileArtifactValue.createDirectory(-1));

    expander = new SpawnInputExpander(execRoot, /*strict=*/ false);
    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/file"), artifact);
  }

  @Test
  public void testRunfilesTwoFiles() throws Exception {
    Artifact artifact1 =
        new Artifact(
            fs.getPath("/root/dir/file"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Artifact artifact2 =
        new Artifact(
            fs.getPath("/root/dir/baz"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Runfiles runfiles = new Runfiles.Builder("workspace")
        .addArtifact(artifact1)
        .addArtifact(artifact2)
        .build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    FakeActionInputFileCache mockCache = new FakeActionInputFileCache();
    mockCache.put(artifact1, FileArtifactValue.createNormalFile(FAKE_DIGEST, 1));
    mockCache.put(artifact2, FileArtifactValue.createNormalFile(FAKE_DIGEST, 2));

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/file"), artifact1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/dir/baz"), artifact2);
  }

  @Test
  public void testRunfilesSymlink() throws Exception {
    Artifact artifact =
        new Artifact(
            fs.getPath("/root/dir/file"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Runfiles runfiles = new Runfiles.Builder("workspace")
        .addSymlink(PathFragment.create("symlink"), artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    FakeActionInputFileCache mockCache = new FakeActionInputFileCache();
    mockCache.put(artifact, FileArtifactValue.createNormalFile(FAKE_DIGEST, 1));

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("runfiles/workspace/symlink"), artifact);
  }

  @Test
  public void testRunfilesRootSymlink() throws Exception {
    Artifact artifact =
        new Artifact(
            fs.getPath("/root/dir/file"),
            ArtifactRoot.asSourceRoot(Root.fromPath(fs.getPath("/root"))));
    Runfiles runfiles = new Runfiles.Builder("workspace")
        .addRootSymlink(PathFragment.create("symlink"), artifact).build();
    RunfilesSupplier supplier = new RunfilesSupplierImpl(PathFragment.create("runfiles"), runfiles);
    FakeActionInputFileCache mockCache = new FakeActionInputFileCache();
    mockCache.put(artifact, FileArtifactValue.createNormalFile(FAKE_DIGEST, 1));

    expander.addRunfilesToInputs(inputMappings, supplier, mockCache);
    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings).containsEntry(PathFragment.create("runfiles/symlink"), artifact);
    // If there's no other entry, Runfiles adds an empty file in the workspace to make sure the
    // directory gets created.
    assertThat(inputMappings)
        .containsEntry(
            PathFragment.create("runfiles/workspace/.runfile"), SpawnInputExpander.EMPTY_FILE);
  }

  @Test
  public void testEmptyManifest() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile("/root/out/_foo/MANIFEST");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).isEmpty();
  }

  @Test
  public void testManifestWithSingleFile() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /dir/file",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("out/foo/bar"), ActionInputHelper.fromPath("/dir/file"));
  }

  @Test
  public void testManifestWithTwoFiles() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /dir/file",
        "<some digest>",
        "workspace/baz /dir/file",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).hasSize(2);
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("out/foo/bar"), ActionInputHelper.fromPath("/dir/file"));
    assertThat(inputMappings)
        .containsEntry(PathFragment.create("out/foo/baz"), ActionInputHelper.fromPath("/dir/file"));
  }

  @Test
  public void testManifestWithDirectory() throws Exception {
    // See AnalysisUtils for the mapping from "foo" to "_foo/MANIFEST".
    scratchFile(
        "/root/out/_foo/MANIFEST",
        "workspace/bar /some",
        "<some digest>");

    ArtifactRoot outputRoot =
        ArtifactRoot.asDerivedRoot(fs.getPath("/root"), fs.getPath("/root/out"));
    Artifact artifact = new Artifact(fs.getPath("/root/out/foo"), outputRoot);
    expander.parseFilesetManifest(inputMappings, artifact, "workspace");
    assertThat(inputMappings).hasSize(1);
    assertThat(inputMappings)
        .containsEntry(
            PathFragment.create("out/foo/bar"), ActionInputHelper.fromPath("/some"));
  }

  private FilesetOutputSymlink filesetSymlink(String from, String to) {
    return new FilesetOutputSymlink(PathFragment.create(from), PathFragment.create(to));
  }

  private ImmutableMap<PathFragment, ImmutableList<FilesetOutputSymlink>> simpleFilesetManifest() {
    return ImmutableMap.of(
        PathFragment.create("out"),
        ImmutableList.of(
            filesetSymlink("workspace/bar", "foo"),
            filesetSymlink("workspace/foo", "/foo/bar")));
  }

  @Test
  public void testManifestWithErrorOnRelativeSymlink() throws Exception {
    expander = new SpawnInputExpander(execRoot, /*strict=*/ true, ERROR);
    try {
      expander.addFilesetManifests(simpleFilesetManifest(), new HashMap<>());
      fail();
    } catch (IOException e) {
      assertThat(e).hasMessageThat().contains("runfiles target is not absolute: foo");
    }
  }

  @Test
  public void testManifestWithIgnoredRelativeSymlink() throws Exception {
    expander = new SpawnInputExpander(execRoot, /*strict=*/ true, IGNORE);
    Map<PathFragment, ActionInput> entries = new HashMap<>();
    expander.addFilesetManifests(simpleFilesetManifest(), entries);
    assertThat(entries)
        .containsExactly(
            PathFragment.create("out/workspace/foo"), ActionInputHelper.fromPath("/foo/bar"));
  }

  @Test
  public void testManifestWithResolvedRelativeSymlink() throws Exception {
    expander = new SpawnInputExpander(execRoot, /*strict=*/ true, RESOLVE);
    Map<PathFragment, ActionInput> entries = new HashMap<>();
    expander.addFilesetManifests(simpleFilesetManifest(), entries);
    assertThat(entries)
        .containsExactly(
            PathFragment.create("out/workspace/bar"), ActionInputHelper.fromPath("/foo/bar"),
            PathFragment.create("out/workspace/foo"), ActionInputHelper.fromPath("/foo/bar"));
  }
}
