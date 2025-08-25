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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.NodeState;
import com.google.devtools.build.skyframe.GraphInconsistencyReceiver.Inconsistency;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.EnqueueParentBehavior;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import java.time.Duration;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Defines the evaluation action used in the multi-threaded Skyframe evaluation, and constructs the
 * {@link ParallelEvaluatorContext} that the actions rely on.
 *
 * <p>This does not implement other parts of Skyframe evaluation setup and post-processing, such as
 * translating a set of requested top-level nodes into actions, or constructing an evaluation
 * result. Derived classes should do this.
 */
public abstract class AbstractParallelEvaluator {
  private static final Logger logger = Logger.getLogger(AbstractParallelEvaluator.class.getName());

  final ProcessableGraph graph;
  final ParallelEvaluatorContext evaluatorContext;
  protected final CycleDetector cycleDetector;

  AbstractParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      int threadCount,
      DirtyTrackingProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      CycleDetector cycleDetector) {
    this.graph = graph;
    this.cycleDetector = cycleDetector;
    evaluatorContext =
        new ParallelEvaluatorContext(
            graph,
            graphVersion,
            skyFunctions,
            reporter,
            emittedEventState,
            keepGoing,
            progressReceiver,
            storedEventFilter,
            errorInfoManager,
            Evaluate::new,
            graphInconsistencyReceiver,
            threadCount);
  }

  AbstractParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      ForkJoinPool forkJoinPool,
      CycleDetector cycleDetector) {
    this.graph = graph;
    this.cycleDetector = cycleDetector;
    evaluatorContext =
        new ParallelEvaluatorContext(
            graph,
            graphVersion,
            skyFunctions,
            reporter,
            emittedEventState,
            keepGoing,
            progressReceiver,
            storedEventFilter,
            errorInfoManager,
            Evaluate::new,
            graphInconsistencyReceiver,
            Preconditions.checkNotNull(forkJoinPool));
  }

  /**
   * If the entry is dirty and not already rebuilding, puts it in a state so that it can rebuild.
   */
  static void maybeMarkRebuilding(NodeEntry entry) {
    if (entry.isDirty() && entry.getDirtyState() != DirtyState.REBUILDING) {
      entry.markRebuilding();
    }
  }

  enum DirtyOutcome {
    ALREADY_PROCESSED,
    NEEDS_EVALUATION
  }

  /** An action that evaluates a value. */
  private class Evaluate implements Runnable {
    /** The name of the value to be evaluated. */
    private final SkyKey skyKey;

    private Evaluate(SkyKey skyKey) {
      this.skyKey = skyKey;
    }

    private void enqueueChild(
        SkyKey skyKey,
        NodeEntry entry,
        SkyKey child,
        NodeEntry childEntry,
        boolean depAlreadyExists)
        throws InterruptedException {
      Preconditions.checkState(!entry.isDone(), "%s %s", skyKey, entry);
      DependencyState dependencyState =
          depAlreadyExists
              ? childEntry.checkIfDoneForDirtyReverseDep(skyKey)
              : childEntry.addReverseDepAndCheckIfDone(skyKey);
      switch (dependencyState) {
        case DONE:
          if (entry.signalDep(childEntry.getVersion())) {
            // This can only happen if there are no more children to be added.
            evaluatorContext.getVisitor().enqueueEvaluation(skyKey);
          }
          break;
        case ALREADY_EVALUATING:
          break;
        case NEEDS_SCHEDULING:
          evaluatorContext.getVisitor().enqueueEvaluation(child);
          break;
      }
    }

    /**
     * Returns true if this depGroup consists of the error transience value and the error transience
     * value is newer than the entry, meaning that the entry must be re-evaluated.
     */
    private boolean invalidatedByErrorTransience(Collection<SkyKey> depGroup, NodeEntry entry)
        throws InterruptedException {
      return depGroup.size() == 1
          && depGroup.contains(ErrorTransienceValue.KEY)
          && !graph.get(
          null, Reason.OTHER, ErrorTransienceValue.KEY).getVersion().atMost(entry.getVersion());
    }

    private DirtyOutcome maybeHandleDirtyNode(NodeEntry state) throws InterruptedException {
      if (!state.isDirty()) {
        return DirtyOutcome.NEEDS_EVALUATION;
      }
      while (state.getDirtyState().equals(DirtyState.CHECK_DEPENDENCIES)) {
        // Evaluating a dirty node for the first time, and checking its children to see if any
        // of them have changed. Note that there must be dirty children for this to happen.

        // Check the children group by group -- we don't want to evaluate a value that is no
        // longer needed because an earlier dependency changed. For example, //foo:foo depends
        // on target //bar:bar and is built. Then foo/BUILD is modified to remove the dependence
        // on bar, and bar/BUILD is deleted. Reloading //bar:bar would incorrectly throw an
        // exception. To avoid this, we must reload foo/BUILD first, at which point we will
        // discover that it has changed, and re-evaluate target //foo:foo from scratch.
        // On the other hand, when an action requests all of its inputs, we can safely check all
        // of them in parallel on a subsequent build. So we allow checking an entire group in
        // parallel here, if the node builder requested a group last build.
        // Note: every dep returned here must either have this node re-registered for it (using
        // checkIfDoneForDirtyReverseDep) and be registered as a direct dep of this node, or have
        // its reverse dep on this node removed. Failing to do either one of these would result in
        // a graph inconsistency, where the child had a reverse dep on this node, but this node
        // had no kind of dependency on the child.
        Collection<SkyKey> directDepsToCheck = state.getNextDirtyDirectDeps();

        if (invalidatedByErrorTransience(directDepsToCheck, state)) {
          // If this dep is the ErrorTransienceValue and the ErrorTransienceValue has been
          // updated then we need to force a rebuild. We would like to just signal the entry as
          // usual, but we can't, because then the ErrorTransienceValue would remain as a dep,
          // which would be incorrect if, for instance, the value re-evaluated to a non-error.
          state.forceRebuild();
          graph.get(skyKey, Reason.RDEP_REMOVAL, ErrorTransienceValue.KEY).removeReverseDep(skyKey);
          return DirtyOutcome.NEEDS_EVALUATION;
        }
        if (!evaluatorContext.keepGoing()) {
          // This check ensures that we maintain the invariant that if a node with an error is
          // reached during a no-keep-going build, none of its currently building parents
          // finishes building. If the child isn't done building yet, it will detect on its own
          // that it has an error (see the VERIFIED_CLEAN case below). On the other hand, if it
          // is done, then it is the parent's responsibility to notice that, which we do here.
          // We check the deps for errors so that we don't continue building this node if it has
          // a child error.
          Map<SkyKey, ? extends NodeEntry> entriesToCheck =
              graph.getBatch(skyKey, Reason.OTHER, directDepsToCheck);
          for (Map.Entry<SkyKey, ? extends NodeEntry> entry : entriesToCheck.entrySet()) {
            if (entry.getValue().isDone() && entry.getValue().getErrorInfo() != null) {
              // If any child has an error, we arbitrarily add a dep on the first one (needed
              // for error bubbling) and throw an exception coming from it.
              SkyKey errorKey = entry.getKey();
              NodeEntry errorEntry = entry.getValue();
              state.addTemporaryDirectDeps(GroupedListHelper.create(errorKey));
              errorEntry.checkIfDoneForDirtyReverseDep(skyKey);
              // Perform the necessary bookkeeping for any deps that are not being used.
              for (Map.Entry<SkyKey, ? extends NodeEntry> depEntry : entriesToCheck.entrySet()) {
                if (!depEntry.getKey().equals(errorKey)) {
                  depEntry.getValue().removeReverseDep(skyKey);
                }
              }
              if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
                // An error was already thrown in the evaluator. Don't do anything here.
                return DirtyOutcome.ALREADY_PROCESSED;
              }
              throw SchedulerException.ofError(
                  errorEntry.getErrorInfo(), entry.getKey(), ImmutableSet.of(skyKey));
            }
          }
        }
        // It is safe to add these deps back to the node -- even if one of them has changed, the
        // contract of pruning is that the node will request these deps again when it rebuilds.
        // We must add these deps before enqueuing them, so that the node knows that it depends
        // on them. If one of these deps is the error transience node, the check we did above
        // in #invalidatedByErrorTransience means that the error transience node is not newer
        // than this node, so we are going to mark it clean (since the error transience node is
        // always the last dep).
        state.addTemporaryDirectDepsGroupToDirtyEntry(directDepsToCheck);
        DepsReport depsReport = graph.analyzeDepsDoneness(skyKey, directDepsToCheck);
        Collection<SkyKey> unknownStatusDeps =
            depsReport.hasInformation() ? depsReport : directDepsToCheck;
        boolean needsScheduling = false;
        for (int i = 0; i < directDepsToCheck.size() - unknownStatusDeps.size(); i++) {
          // Since all of these nodes were done at an earlier version than this one, we may safely
          // signal with the minimal version, since they cannot trigger a re-evaluation.
          needsScheduling = state.signalDep(MinimalVersion.INSTANCE);
        }
        if (needsScheduling) {
          Preconditions.checkState(
              unknownStatusDeps.isEmpty(),
              "Ready without all deps checked? %s %s %s",
              skyKey,
              state,
              unknownStatusDeps);
          continue;
        }
        handleKnownChildrenForDirtyNode(unknownStatusDeps, state);
        return DirtyOutcome.ALREADY_PROCESSED;
      }
      switch (state.getDirtyState()) {
        case VERIFIED_CLEAN:
          // No child has a changed value. This node can be marked done and its parents signaled
          // without any re-evaluation.
          Set<SkyKey> reverseDeps = state.markClean();
          // Tell the receiver that the value was not actually changed this run.
          evaluatorContext
              .getProgressReceiver()
              .evaluated(skyKey, new SkyValueSupplier(state), EvaluationState.CLEAN);
          if (!evaluatorContext.keepGoing() && state.getErrorInfo() != null) {
            if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
              return DirtyOutcome.ALREADY_PROCESSED;
            }
            throw SchedulerException.ofError(state.getErrorInfo(), skyKey, reverseDeps);
          }
          evaluatorContext.signalValuesAndEnqueueIfReady(
              skyKey, reverseDeps, state.getVersion(), EnqueueParentBehavior.ENQUEUE);
          return DirtyOutcome.ALREADY_PROCESSED;
        case NEEDS_REBUILDING:
          maybeMarkRebuilding(state);
          // Fall through to REBUILDING case.
        case REBUILDING:
          return DirtyOutcome.NEEDS_EVALUATION;
        default:
          throw new IllegalStateException("key: " + skyKey + ", entry: " + state);
      }
    }

    private void handleKnownChildrenForDirtyNode(Collection<SkyKey> knownChildren, NodeEntry state)
        throws InterruptedException {
      Map<SkyKey, ? extends NodeEntry> oldChildren =
          graph.getBatch(skyKey, Reason.ENQUEUING_CHILD, knownChildren);
      if (oldChildren.size() != knownChildren.size()) {
        GraphInconsistencyReceiver inconsistencyReceiver =
            evaluatorContext.getGraphInconsistencyReceiver();
        Set<SkyKey> missingChildren =
            Sets.difference(ImmutableSet.copyOf(knownChildren), oldChildren.keySet());
        for (SkyKey missingChild : missingChildren) {
          inconsistencyReceiver.noteInconsistencyAndMaybeThrow(
              skyKey, missingChild, Inconsistency.CHILD_MISSING_FOR_DIRTY_NODE);
        }
        Map<SkyKey, ? extends NodeEntry> recreatedEntries =
            graph.createIfAbsentBatch(skyKey, Reason.ENQUEUING_CHILD, missingChildren);
        for (Map.Entry<SkyKey, ? extends NodeEntry> recreatedEntry : recreatedEntries.entrySet()) {
          enqueueChild(
              skyKey,
              state,
              recreatedEntry.getKey(),
              recreatedEntry.getValue(),
              /*depAlreadyExists=*/ false);
        }
      }
      for (Map.Entry<SkyKey, ? extends NodeEntry> e : oldChildren.entrySet()) {
        SkyKey directDep = e.getKey();
        NodeEntry directDepEntry = e.getValue();
        // TODO(bazel-team): If this signals the current node and makes it ready, consider
        // evaluating it in this thread instead of scheduling a new evaluation.
        enqueueChild(skyKey, state, directDep, directDepEntry, /*depAlreadyExists=*/ true);
      }
    }

    @Override
    public void run() {
      try {
        NodeEntry state =
            Preconditions.checkNotNull(graph.get(null, Reason.EVALUATION, skyKey), skyKey);
        Preconditions.checkState(state.isReady(), "%s %s", skyKey, state);
        try {
          evaluatorContext.getProgressReceiver().stateStarting(skyKey, NodeState.CHECK_DIRTY);
          if (maybeHandleDirtyNode(state) == DirtyOutcome.ALREADY_PROCESSED) {
            return;
          }
        } finally {
          evaluatorContext.getProgressReceiver().stateEnding(skyKey, NodeState.CHECK_DIRTY, -1);
        }

        Set<SkyKey> oldDeps = state.getAllRemainingDirtyDirectDeps();
        SkyFunctionEnvironment env;
        try {
          evaluatorContext.getProgressReceiver().stateStarting(skyKey,
              NodeState.INITIALIZING_ENVIRONMENT);
          env = new SkyFunctionEnvironment(skyKey, state.getTemporaryDirectDeps(), oldDeps,
              evaluatorContext);
        } finally {
          evaluatorContext.getProgressReceiver().stateEnding(skyKey,
              NodeState.INITIALIZING_ENVIRONMENT, -1);
        }
        SkyFunctionName functionName = skyKey.functionName();
        SkyFunction factory =
            Preconditions.checkNotNull(
                evaluatorContext.getSkyFunctions().get(functionName),
                "Unable to find SkyFunction '%s' for node with key %s, %s",
                functionName,
                skyKey,
                state);

        SkyValue value = null;
        long startTime = BlazeClock.instance().nanoTime();
        try {
          try {
            evaluatorContext.getProgressReceiver().stateStarting(skyKey, NodeState.COMPUTE);
            value = factory.compute(skyKey, env);
          } finally {
            long elapsedTimeNanos = BlazeClock.instance().nanoTime() - startTime;
            if (elapsedTimeNanos > 0) {
              evaluatorContext.getProgressReceiver().stateEnding(skyKey, NodeState.COMPUTE,
                  elapsedTimeNanos);
              Profiler.instance()
                  .logSimpleTaskDuration(
                      startTime,
                      Duration.ofNanos(elapsedTimeNanos),
                      ProfilerTask.SKYFUNCTION,
                      skyKey.functionName().getName());
            }
          }
        } catch (final SkyFunctionException builderException) {
          ReifiedSkyFunctionException reifiedBuilderException =
              new ReifiedSkyFunctionException(builderException, skyKey);
          // In keep-going mode, we do not let SkyFunctions throw errors with missing deps -- we
          // will restart them when their deps are done, so we can have a definitive error and
          // definitive graph structure, thus avoiding non-determinism. It's completely reasonable
          // for SkyFunctions to throw eagerly because they do not know if they are in keep-going
          // mode.
          // Propagated transitive errors are treated the same as missing deps.
          if ((!evaluatorContext.keepGoing() || !env.valuesMissing())
              && reifiedBuilderException.getRootCauseSkyKey().equals(skyKey)) {
            boolean shouldFailFast =
                !evaluatorContext.keepGoing() || builderException.isCatastrophic();
            if (shouldFailFast) {
              // After we commit this error to the graph but before the doMutatingEvaluation call
              // completes with the error there is a race-like opportunity for the error to be used,
              // either by an in-flight computation or by a future computation.
              if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
                // This is not the first error encountered, so we ignore it so that we can terminate
                // with the first error.
                return;
              } else {
                logger.warning(
                    "Aborting evaluation due to "
                        + builderException
                        + " while evaluating "
                        + skyKey);
              }
            }

            Map<SkyKey, ? extends NodeEntry> newlyRequestedDeps =
                evaluatorContext.getBatchValues(
                    skyKey, Reason.DONE_CHECKING, env.getNewlyRequestedDeps());
            boolean isTransitivelyTransient = reifiedBuilderException.isTransient();
            for (NodeEntry depEntry :
                Iterables.concat(env.getDirectDepsValues(), newlyRequestedDeps.values())) {
              if (!isDoneForBuild(depEntry)) {
                continue;
              }
              ErrorInfo depError = depEntry.getErrorInfo();
              if (depError != null) {
                isTransitivelyTransient |= depError.isTransitivelyTransient();
              }
            }
            ErrorInfo errorInfo = evaluatorContext.getErrorInfoManager().fromException(
                skyKey,
                reifiedBuilderException,
                isTransitivelyTransient);
            registerNewlyDiscoveredDepsForDoneEntry(
                skyKey, state, oldDeps, env, evaluatorContext.keepGoing());
            env.setError(state, errorInfo);
            Set<SkyKey> rdepsToBubbleUpTo =
                env.commit(
                    state,
                    shouldFailFast ? EnqueueParentBehavior.SIGNAL : EnqueueParentBehavior.ENQUEUE);
            if (!shouldFailFast) {
              return;
            }
            throw SchedulerException.ofError(errorInfo, skyKey, rdepsToBubbleUpTo);
          }
        } catch (RuntimeException re) {
          // Programmer error (most likely NPE or a failed precondition in a SkyFunction). Output
          // some context together with the exception.
          String msg = prepareCrashMessage(skyKey, state.getInProgressReverseDeps());
          RuntimeException ex = new RuntimeException(msg, re);
          evaluatorContext.getVisitor().noteCrash(ex);
          throw ex;
        } finally {
          env.doneBuilding();
        }

        if (maybeEraseNodeToRestartFromScratch(skyKey, state, value)) {
          evaluatorContext.getVisitor().enqueueEvaluation(skyKey);
          return;
        }

        // Helper objects for all the newly requested deps that weren't known to the environment,
        // and may contain duplicate elements.
        GroupedListHelper<SkyKey> newDirectDeps = env.getNewlyRequestedDeps();

        if (value != null) {
          Preconditions.checkState(
              !env.valuesMissing(),
              "Evaluation of %s returned non-null value but requested dependencies that weren't "
                  + "computed yet (one of %s), NodeEntry: %s",
              skyKey,
              newDirectDeps,
              state);

          try {
            evaluatorContext.getProgressReceiver().stateStarting(skyKey, NodeState.COMMIT);
            env.setValue(value);
            registerNewlyDiscoveredDepsForDoneEntry(
                skyKey, state, oldDeps, env, evaluatorContext.keepGoing());
            env.commit(state, EnqueueParentBehavior.ENQUEUE);
          } finally {
            evaluatorContext.getProgressReceiver().stateEnding(skyKey, NodeState.COMMIT, -1);
          }
          return;
        }

        if (env.getDepErrorKey() != null) {
          Preconditions.checkState(
              !evaluatorContext.keepGoing(), "%s %s %s", skyKey, state, env.getDepErrorKey());
          // We encountered a child error in noKeepGoing mode, so we want to fail fast. But we first
          // need to add the edge between the current node and the child error it requested so that
          // error bubbling can occur. Note that this edge will subsequently be removed during graph
          // cleaning (since the current node will never be committed to the graph).
          SkyKey childErrorKey = env.getDepErrorKey();
          NodeEntry childErrorEntry =
              Preconditions.checkNotNull(
                  graph.get(skyKey, Reason.OTHER, childErrorKey),
                  "skyKey: %s, state: %s childErrorKey: %s",
                  skyKey,
                  state,
                  childErrorKey);
          if (newDirectDeps.contains(childErrorKey)) {
            // Add this dep if it was just requested. In certain rare race conditions (see
            // MemoizingEvaluatorTest.cachedErrorCausesRestart) this dep may have already been
            // requested.
            state.addTemporaryDirectDeps(GroupedListHelper.create(childErrorKey));
            DependencyState childErrorState;
            if (oldDeps.contains(childErrorKey)) {
              childErrorState = childErrorEntry.checkIfDoneForDirtyReverseDep(skyKey);
            } else {
              childErrorState = childErrorEntry.addReverseDepAndCheckIfDone(skyKey);
            }
            Preconditions.checkState(
                childErrorState == DependencyState.DONE,
                "skyKey: %s, state: %s childErrorKey: %s",
                skyKey,
                state,
                childErrorKey,
                childErrorEntry);
          }
          ErrorInfo childErrorInfo = Preconditions.checkNotNull(childErrorEntry.getErrorInfo());
          evaluatorContext.getVisitor().preventNewEvaluations();
          throw SchedulerException.ofError(childErrorInfo, childErrorKey, ImmutableSet.of(skyKey));
        }

        // TODO(bazel-team): This code is not safe to interrupt, because we would lose the state in
        // newDirectDeps.

        // TODO(bazel-team): An ill-behaved SkyFunction can throw us into an infinite loop where we
        // add more dependencies on every run. [skyframe-core]

        // Add all the newly requested dependencies to the temporary direct deps. Note that
        // newDirectDeps does not contain any elements in common with the already existing temporary
        // direct deps. uniqueNewDeps will be the set of unique keys contained in newDirectDeps.
        Set<SkyKey> uniqueNewDeps = state.addTemporaryDirectDeps(newDirectDeps);

        // If there were no newly requested dependencies, at least one of them was in error or there
        // is a bug in the SkyFunction implementation. The environment has collected its errors, so
        // we just order it to be built.
        if (uniqueNewDeps.isEmpty()) {
          // TODO(bazel-team): This means a bug in the SkyFunction. What to do?
          Preconditions.checkState(
              !env.getChildErrorInfos().isEmpty(),
              "Evaluation of SkyKey failed and no dependencies were requested: %s %s",
              skyKey,
              state);
          Preconditions.checkState(
              evaluatorContext.keepGoing(),
              "nokeep_going evaluation should have failed on first child error: %s %s %s",
              skyKey,
              state,
              env.getChildErrorInfos());
          // If the child error was catastrophic, committing this parent to the graph is not
          // necessary, but since we don't do error bubbling in catastrophes, it doesn't violate any
          // invariants either.
          env.commit(state, EnqueueParentBehavior.ENQUEUE);
          return;
        }

        // We want to split apart the dependencies that existed for this node the last time we did
        // an evaluation and those that were introduced in this evaluation. To be clear, the prefix
        // "newDeps" refers to newly discovered this time around after a SkyFunction#compute call
        // and not to be confused with the oldDeps variable which refers to the last evaluation,
        // (ie) a prior call to ParallelEvaluator#eval).
        Set<SkyKey> newDepsThatWerentInTheLastEvaluation = Sets.difference(uniqueNewDeps, oldDeps);
        Set<SkyKey> newDepsThatWereInTheLastEvaluation =
            Sets.difference(uniqueNewDeps, newDepsThatWerentInTheLastEvaluation);

        InterruptibleSupplier<Map<SkyKey, ? extends NodeEntry>>
            newDepsThatWerentInTheLastEvaluationNodes =
                graph.createIfAbsentBatchAsync(
                    skyKey, Reason.RDEP_ADDITION, newDepsThatWerentInTheLastEvaluation);
        handleKnownChildrenForDirtyNode(newDepsThatWereInTheLastEvaluation, state);

        for (Map.Entry<SkyKey, ? extends NodeEntry> e :
            newDepsThatWerentInTheLastEvaluationNodes.get().entrySet()) {
          SkyKey newDirectDep = e.getKey();
          NodeEntry newDirectDepEntry = e.getValue();
          enqueueChild(skyKey, state, newDirectDep, newDirectDepEntry, /*depAlreadyExists=*/ false);
        }
        // It is critical that there is no code below this point in the try block.
      } catch (InterruptedException ie) {
        // InterruptedException cannot be thrown by Runnable.run, so we must wrap it.
        // Interrupts can be caught by both the Evaluator and the AbstractQueueVisitor.
        // The former will unwrap the IE and propagate it as is; the latter will throw a new IE.
        throw SchedulerException.ofInterruption(ie, skyKey);
      }
    }

    private String prepareCrashMessage(SkyKey skyKey, Iterable<SkyKey> reverseDeps) {
      StringBuilder reverseDepDump = new StringBuilder();
      for (SkyKey key : reverseDeps) {
        if (reverseDepDump.length() > MAX_REVERSEDEP_DUMP_LENGTH) {
          reverseDepDump.append(", ...");
          break;
        }
        if (reverseDepDump.length() > 0) {
          reverseDepDump.append(", ");
        }
        reverseDepDump.append("'");
        reverseDepDump.append(key.toString());
        reverseDepDump.append("'");
      }

      return String.format(
          "Unrecoverable error while evaluating node '%s' (requested by nodes %s)",
          skyKey, reverseDepDump);
    }

    private static final int MAX_REVERSEDEP_DUMP_LENGTH = 1000;
  }

  private boolean maybeEraseNodeToRestartFromScratch(
      SkyKey key, NodeEntry entry, SkyValue returnedValue) {
    if (!SkyFunction.SENTINEL_FOR_RESTART_FROM_SCRATCH.equals(returnedValue)) {
      return false;
    }
    evaluatorContext
        .getGraphInconsistencyReceiver()
        .noteInconsistencyAndMaybeThrow(key, /*otherKey=*/ null, Inconsistency.RESET_REQUESTED);
    entry.resetForRestartFromScratch();
    // TODO(mschaller): rdeps of children have to be handled here. If the graph does not keep edges,
    // nothing has to be done, since there are no reverse deps to keep consistent. If the graph
    // keeps edges, it's a harder problem. The reverse deps could just be removed, but in the case
    // that this node is dirty, the deps shouldn't be removed, they should just be transformed back
    // to "known reverse deps" from "reverse deps declared during this evaluation" (the inverse of
    // NodeEntry#checkIfDoneForDirtyReverseDep). Such a method doesn't currently exist, but could.
    return true;
  }

  void propagateEvaluatorContextCrashIfAny() {
    if (!evaluatorContext.getVisitor().getCrashes().isEmpty()) {
      evaluatorContext
          .getReporter()
          .handle(Event.error("Crashes detected: " + evaluatorContext.getVisitor().getCrashes()));
      throw Preconditions.checkNotNull(
          Iterables.getFirst(evaluatorContext.getVisitor().getCrashes(), null));
    }
  }

  void propagateInterruption(SchedulerException e) throws InterruptedException {
    boolean mustThrowInterrupt = Thread.interrupted();
    Throwables.propagateIfPossible(e.getCause(), InterruptedException.class);
    if (mustThrowInterrupt) {
      // As per the contract of AbstractQueueVisitor#work, if an unchecked exception is thrown and
      // the build is interrupted, the thrown exception is what will be rethrown. Since the user
      // presumably wanted to interrupt the build, we ignore the thrown SchedulerException (which
      // doesn't indicate a programming bug) and throw an InterruptedException.
      throw new InterruptedException();
    }
  }

  /**
   * Add any additional deps that were registered during the run of a builder that finished by
   * creating a node or throwing an error. Builders may throw errors even if all their deps were not
   * provided -- we trust that a SkyFunction might know it should throw an error even if not all of
   * its requested deps are done. However, that means we're assuming the SkyFunction would throw
   * that same error if all of its requested deps were done. Unfortunately, there is no way to
   * enforce that condition.
   *
   * @throws InterruptedException
   */
  private void registerNewlyDiscoveredDepsForDoneEntry(
      SkyKey skyKey,
      NodeEntry entry,
      Set<SkyKey> oldDeps,
      SkyFunctionEnvironment env,
      boolean keepGoing)
      throws InterruptedException {

    // We don't expect any unfinished deps in a keep-going build.
    if (!keepGoing) {
      Map<SkyKey, ? extends NodeEntry> newlyRequestedDepMap =
          graph.getBatch(skyKey, Reason.DONE_CHECKING, env.getNewlyRequestedDeps());
      Iterator<SkyKey> it = env.getNewlyRequestedDeps().iterator();
      if (!it.hasNext()) {
        return;
      }
      Set<SkyKey> unfinishedDeps = new HashSet<>();
      while (it.hasNext()) {
        SkyKey dep = it.next();
        if (!isDoneForBuild(newlyRequestedDepMap.get(dep))) {
          unfinishedDeps.add(dep);
        }
      }
      env.getNewlyRequestedDeps().remove(unfinishedDeps);
    }

    Set<SkyKey> uniqueNewDeps = entry.addTemporaryDirectDeps(env.getNewlyRequestedDeps());
    Set<SkyKey> newlyAddedNewDeps = Sets.difference(uniqueNewDeps, oldDeps);
    Set<SkyKey> previouslyRegisteredNewDeps = Sets.difference(uniqueNewDeps, newlyAddedNewDeps);

    InterruptibleSupplier<Map<SkyKey, ? extends NodeEntry>> newlyAddedNewDepNodes =
        graph.getBatchAsync(skyKey, Reason.RDEP_ADDITION, newlyAddedNewDeps);

    for (Map.Entry<SkyKey, ? extends NodeEntry> newDep :
        graph.getBatch(skyKey, Reason.SIGNAL_DEP, previouslyRegisteredNewDeps).entrySet()) {
      // Note that this depEntry can't be null. In a keep-going build, we expect all deps to be
      // done. In a non-keep-going build, If env.newlyRequestedDeps contained a key with a
      // null entry, then it would have been added to unfinishedDeps and then removed from
      // env.newlyRequestedDeps just above this loop.
      NodeEntry depEntry = newDep.getValue();
      DependencyState triState = depEntry.checkIfDoneForDirtyReverseDep(skyKey);
      Preconditions.checkState(
          DependencyState.DONE == triState,
          "new dep %s was not already done for %s. NodeEntry: %s. DepNodeEntry: %s",
          newDep,
          skyKey,
          entry,
          depEntry);
      entry.signalDep();
    }

    for (SkyKey newDep : newlyAddedNewDeps) {
      // Note that this depEntry can't be null. In a keep-going build, we expect all deps to be
      // done. In a non-keep-going build, If env.newlyRequestedDeps contained a key with a
      // null entry, then it would have been added to unfinishedDeps and then removed from
      // env.newlyRequestedDeps just above this loop.
      NodeEntry depEntry =
          Preconditions.checkNotNull(newlyAddedNewDepNodes.get().get(newDep), newDep);
      DependencyState triState = depEntry.addReverseDepAndCheckIfDone(skyKey);
      Preconditions.checkState(
          DependencyState.DONE == triState,
          "new dep %s was not already done for %s. NodeEntry: %s. DepNodeEntry: %s",
          newDep,
          skyKey,
          entry,
          depEntry);
      entry.signalDep();
    }
    Preconditions.checkState(
        entry.isReady(), "%s %s %s", skyKey, entry, env.getNewlyRequestedDeps());
  }

  /**
   * Return true if the entry does not need to be re-evaluated this build. The entry will need to be
   * re-evaluated if it is not done, but also if it was not completely evaluated last build and this
   * build is keepGoing.
   */
  static boolean isDoneForBuild(@Nullable NodeEntry entry) {
    return entry != null && entry.isDone();
  }
}
