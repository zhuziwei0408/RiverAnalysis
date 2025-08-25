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

package com.google.devtools.build.lib.buildeventservice;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Strings.isNullOrEmpty;
import static com.google.devtools.build.lib.buildeventservice.BuildEventServiceTransport.UPLOAD_FAILED_MESSAGE;
import static java.lang.String.format;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.buildeventservice.client.BuildEventServiceClient;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.PathConverter;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventStreamOptions;
import com.google.devtools.build.lib.buildeventstream.transports.BuildEventTransportFactory;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.BuildEventStreamer;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.SynchronizedOutputStream;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsProvider;
import java.io.IOException;
import java.util.Set;
import java.util.logging.Logger;
import javax.annotation.Nullable;

/**
 * Module responsible for the Build Event Transport (BEP) and Build Event Service (BES)
 * functionality.
 */
public abstract class BuildEventServiceModule<T extends BuildEventServiceOptions>
    extends BlazeModule {

  private static final Logger logger = Logger.getLogger(BuildEventServiceModule.class.getName());

  private OutErr outErr;

  private Set<BuildEventTransport> transports = ImmutableSet.of();

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(
        optionsClass(),
        AuthAndTLSOptions.class,
        BuildEventStreamOptions.class,
        BuildEventProtocolOptions.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment commandEnvironment)
      throws AbruptExitException {
    // Reset to null in case afterCommand was not called.
    this.outErr = null;
    if (!whitelistedCommands().contains(commandEnvironment.getCommandName())) {
      return;
    }

    BuildEventStreamer streamer =
        tryCreateStreamer(
            commandEnvironment.getRuntime().getStartupOptionsProvider(),
            commandEnvironment.getOptions(),
            commandEnvironment.getReporter(),
            commandEnvironment.getBlazeModuleEnvironment(),
            commandEnvironment.getRuntime().getClock(),
            commandEnvironment.getRuntime().getPathToUriConverter(),
            commandEnvironment.getReporter(),
            commandEnvironment.getBuildRequestId().toString(),
            commandEnvironment.getCommandId().toString(),
            commandEnvironment.getCommandName());
    if (streamer != null) {
      commandEnvironment.getReporter().addHandler(streamer);
      commandEnvironment.getEventBus().register(streamer);
      long bufferSize =
          commandEnvironment.getOptions().getOptions(optionsClass()).besOuterrBufferSize;

      final SynchronizedOutputStream out = new SynchronizedOutputStream(bufferSize);
      final SynchronizedOutputStream err = new SynchronizedOutputStream(bufferSize);
      this.outErr = OutErr.create(out, err);
      streamer.registerOutErrProvider(
          new BuildEventStreamer.OutErrProvider() {
            @Override
            public String getOut() {
              return out.readAndReset();
            }

            @Override
            public String getErr() {
              return err.readAndReset();
            }
          });
      err.registerStreamer(streamer);
      out.registerStreamer(streamer);
      logger.fine("BuildEventStreamer created and registered successfully.");
    }
  }

  @Override
  public OutErr getOutputListener() {
    return outErr;
  }

  @Override
  public void afterCommand() {
    this.outErr = null;
  }

  /**
   * Returns {@code null} if no stream could be created.
   */
  @Nullable
  @VisibleForTesting
  BuildEventStreamer tryCreateStreamer(
      OptionsProvider startupOptionsProvider,
      OptionsProvider optionsProvider,
      EventHandler commandLineReporter,
      ModuleEnvironment moduleEnvironment,
      Clock clock,
      PathConverter pathConverter,
      Reporter reporter,
      String buildRequestId,
      String invocationId,
      String commandName) {
    try {
      T besOptions =
          checkNotNull(
              optionsProvider.getOptions(optionsClass()),
              "Could not get BuildEventServiceOptions.");
      AuthAndTLSOptions authTlsOptions =
          checkNotNull(optionsProvider.getOptions(AuthAndTLSOptions.class),
              "Could not get AuthAndTLSOptions.");
      BuildEventStreamOptions bepOptions =
          checkNotNull(optionsProvider.getOptions(BuildEventStreamOptions.class),
          "Could not get BuildEventStreamOptions.");
      BuildEventProtocolOptions protocolOptions =
          checkNotNull(optionsProvider.getOptions(BuildEventProtocolOptions.class),
          "Could not get BuildEventProtocolOptions.");

      BuildEventTransport besTransport = null;
      try {
        besTransport =
            tryCreateBesTransport(
                besOptions,
                authTlsOptions,
                buildRequestId,
                invocationId,
                commandName,
                moduleEnvironment,
                clock,
                protocolOptions,
                pathConverter,
                commandLineReporter,
                startupOptionsProvider);
      } catch (Exception e) {
        if (besOptions.besBestEffort) {
          commandLineReporter.handle(Event.warn(format(UPLOAD_FAILED_MESSAGE, e.getMessage())));
        } else {
          commandLineReporter.handle(Event.error(format(UPLOAD_FAILED_MESSAGE, e.getMessage())));
          moduleEnvironment.exit(new AbruptExitException(
              "Failed while creating BuildEventTransport", ExitCode.PUBLISH_ERROR));
          return null;
        }
      }

      ImmutableSet<BuildEventTransport> bepTransports =
          BuildEventTransportFactory.createFromOptions(bepOptions, protocolOptions, pathConverter);

      ImmutableSet.Builder<BuildEventTransport> transportsBuilder =
          ImmutableSet.<BuildEventTransport>builder().addAll(bepTransports);
      if (besTransport != null) {
        transportsBuilder.add(besTransport);
      }

      transports = transportsBuilder.build();
      if (!transports.isEmpty()) {
        BuildEventStreamOptions buildEventStreamOptions =
            optionsProvider.getOptions(BuildEventStreamOptions.class);
        return new BuildEventStreamer(transports, reporter, buildEventStreamOptions);
      }
    } catch (Exception e) {
      moduleEnvironment.exit(new AbruptExitException(ExitCode.LOCAL_ENVIRONMENTAL_ERROR, e));
    }
    return null;
  }

  @Nullable
  private BuildEventTransport tryCreateBesTransport(
      T besOptions,
      AuthAndTLSOptions authTlsOptions,
      String buildRequestId,
      String invocationId,
      String commandName,
      ModuleEnvironment moduleEnvironment,
      Clock clock,
      BuildEventProtocolOptions protocolOptions,
      PathConverter pathConverter,
      EventHandler commandLineReporter,
      OptionsProvider startupOptionsProvider)
      throws IOException, OptionsParsingException {
    if (isNullOrEmpty(besOptions.besBackend)) {
      logger.fine("BuildEventServiceTransport is disabled.");
      return null;
    } else {
      logger.fine(format("Will create BuildEventServiceTransport streaming to '%s'",
          besOptions.besBackend));

      final String message;
      if (!Strings.isNullOrEmpty(besOptions.besResultsUrl)) {
        String url =
            besOptions.besResultsUrl.endsWith("/")
                ? besOptions.besResultsUrl
                : besOptions.besResultsUrl + "/";
        message = "Streaming Build Event Protocol to " + url + invocationId;
      } else {
        message =
            format(
                "Streaming Build Event Protocol to %s build_request_id: %s " + "invocation_id: %s",
                besOptions.besBackend, buildRequestId, invocationId);
      }
      commandLineReporter.handle(Event.info(message));

      BuildEventTransport besTransport =
          new BuildEventServiceTransport(
              createBesClient(besOptions, authTlsOptions),
              besOptions.besTimeout,
              besOptions.besBestEffort,
              besOptions.besLifecycleEvents,
              buildRequestId,
              invocationId,
              commandName,
              moduleEnvironment,
              clock,
              protocolOptions,
              pathConverter,
              commandLineReporter,
              besOptions.projectId,
              keywords(besOptions, startupOptionsProvider));
      logger.fine("BuildEventServiceTransport was created successfully");
      return besTransport;
    }
  }

  @Override
  public void blazeShutdown() {
    for (BuildEventTransport transport : transports) {
      transport.closeNow();
    }
  }

  protected abstract Class<T> optionsClass();

  protected abstract BuildEventServiceClient createBesClient(
      T besOptions, AuthAndTLSOptions authAndTLSOptions)
      throws IOException, OptionsParsingException;

  protected abstract Set<String> whitelistedCommands();

  protected Set<String> keywords(T besOptions, @Nullable OptionsProvider startupOptionsProvider) {
    return besOptions
        .besKeywords
        .stream()
        .map(keyword -> "user_keyword=" + keyword)
        .collect(ImmutableSet.toImmutableSet());
  }
}
