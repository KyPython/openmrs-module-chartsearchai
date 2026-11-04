/**
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/. OpenMRS is also distributed under
 * the terms of the Healthcare Disclaimer located at http://openmrs.org/license.
 *
 * Copyright (C) OpenMRS Inc. OpenMRS is a registered trademark and the OpenMRS
 * graphic logo is a trademark of OpenMRS Inc.
 */
package org.openmrs.module.chartsearchai.api.impl;

import java.io.IOException;
import java.io.InputStream;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The one place an operator-configured remote response body is read (issue #627). Every
 * {@code HttpResponse.body()} invocation in production lives in the four methods below, and
 * {@code ArchitectureGuardTest.remoteResponseBodyReadsGoThroughOneClass} asks that of the
 * compiled classes — a read reached through a renamed reference is still the same pool
 * entries, which is the half the source-text rule this replaces could not see.
 *
 * <p>Each method pairs extraction with its ceiling or mode, so there is no bare "hand me the
 * stream" for a future caller to read past: the full read aborts past
 * {@link RemoteLlmEngine#MAX_RESPONSE_BYTES}, the error read truncates under
 * {@link RemoteLlmEngine#MAX_ERROR_BODY_BYTES}, the streaming read returns a
 * {@link BoundedResponseStream} whose ceiling travels with the object, and the status-only
 * call closes the body unread (the #620 probe semantics, unchanged).</p>
 *
 * <p>The streaming caller's extra wrap is deliberate, not leftover: the
 * {@code InputStream}-taking parse path keeps its own ceiling for its direct unit-test
 * callers, so a ceilinged stream is wrapped twice — same limit, same throw either way.</p>
 *
 * <p>Out of this class by design: {@code LocalLlmEngine}, whose peer is this module's own
 * subprocess rather than an operator-configured address, and the test tree, where a live
 * endpoint's body is read on purpose. Both exclusions live in the guard test, where they
 * are reviewable rather than merely absent.</p>
 *
 * <p>The error-body debug line moved here with its method; its message is unchanged and its
 * logger category is now this class.</p>
 */
public final class RemoteResponseBodies {

	private static final Logger log = LoggerFactory.getLogger(RemoteResponseBodies.class);

	private RemoteResponseBodies() {
	}

	/**
	 * The whole body as text, or {@link BoundedResponseStream.ResponseTooLargeException} if the
	 * peer sent more than {@link RemoteLlmEngine#MAX_RESPONSE_BYTES}. Aborts rather than
	 * truncating, unlike {@link #readTruncatedError}: half a completion envelope is not a
	 * completion, and a caller handed one would report a parse failure for what is really an
	 * oversized response.
	 */
	public static String readBounded(HttpResponse<InputStream> response) throws IOException {
		try (InputStream ceilinged = new BoundedResponseStream(response.body(),
				RemoteLlmEngine.MAX_RESPONSE_BYTES)) {
			return new String(ceilinged.readAllBytes(), StandardCharsets.UTF_8);
		}
	}

	/**
	 * At most {@link RemoteLlmEngine#MAX_ERROR_BODY_BYTES} of a non-2xx body, abandoning the
	 * rest unread — truncating rather than raising, for the reason that constant states. The
	 * cut is at a byte and not a character boundary, which costs at worst one replacement
	 * character: the result only ever reaches a log line, never a parser.
	 */
	public static String readTruncatedError(HttpResponse<InputStream> response) {
		try (InputStream in = response.body()) {
			// The three-arg form fills a buffer this method sizes. The one-arg readNBytes(int)
			// would allocate its own and then copy out whatever the body actually was, which for
			// a short error body — the common case — is a second allocation for nothing.
			byte[] buffer = new byte[RemoteLlmEngine.MAX_ERROR_BODY_BYTES];
			return new String(buffer, 0, in.readNBytes(buffer, 0, buffer.length),
					StandardCharsets.UTF_8);
		}
		catch (IOException e) {
			log.debug("Could not read the error body from the remote LLM API", e);
			return "";
		}
	}

	/**
	 * The body as a stream that stops instead of growing: at most
	 * {@link RemoteLlmEngine#MAX_RESPONSE_BYTES}, then
	 * {@link BoundedResponseStream.ResponseTooLargeException}. The ceiling travels with the
	 * object, so SSE parsing over the returned stream is bounded whatever the parser does —
	 * and closing it closes the underlying body, which is what cancels the exchange.
	 */
	public static InputStream streamingBody(HttpResponse<InputStream> response) {
		return new BoundedResponseStream(response.body(), RemoteLlmEngine.MAX_RESPONSE_BYTES);
	}

	/**
	 * The status-only call: closes the body unread and reads nothing. The probe that uses it
	 * wants the status code and nothing else, and a peer that answers and holds its body open
	 * must not stall provider discovery.
	 */
	public static void closeWithoutReading(HttpResponse<InputStream> response) throws IOException {
		response.body().close();
	}
}
