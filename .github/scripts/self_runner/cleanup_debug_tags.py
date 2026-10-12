#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Delete a temporary release tag from the internal registry or Docker Hub."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional


INTERNAL_ACCEPT = (
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json, "
    "application/vnd.docker.distribution.manifest.v2+json, */*"
)


def http_request(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[dict[str, str]] = None,
    data: Optional[bytes] = None,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def basic_authorization(username: str, password: str) -> str:
    value = f"{username}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(value).decode("ascii")


def split_image_reference(
    image: str,
    *,
    expected_host: Optional[str] = None,
) -> tuple[Optional[str], str, str]:
    reference = image
    if "://" in reference:
        reference = reference.split("://", 1)[1]

    host: Optional[str] = expected_host
    if host is None and "/" in reference:
        candidate = reference.split("/", 1)[0]
        if "." in candidate or ":" in candidate or candidate == "localhost":
            host = candidate

    if host is not None and reference.startswith(f"{host}/"):
        reference = reference[len(host) + 1 :]

    if ":" not in reference.rsplit("/", 1)[-1]:
        raise ValueError("image reference must include a tag")
    repository, tag = reference.rsplit(":", 1)
    if not repository or not tag:
        raise ValueError("image reference must include a repository and tag")
    return host, repository, tag


def normalize_dockerhub_repository(repository: str) -> str:
    if repository.startswith("docker.io/"):
        return repository[len("docker.io/") :]
    return repository


def delete_internal_registry_tag(
    *,
    host: str,
    image: str,
    username: str,
    password: str,
) -> None:
    resolved_host, repository, tag = split_image_reference(image, expected_host=host)
    if resolved_host is None:
        raise ValueError("internal registry host could not be resolved")

    authorization = basic_authorization(username, password)
    manifest_url = (
        f"https://{resolved_host}/v2/{repository}/manifests/"
        f"{urllib.parse.quote(tag, safe='')}"
    )
    status, headers, _ = http_request(
        manifest_url,
        method="HEAD",
        headers={"Accept": INTERNAL_ACCEPT, "Authorization": authorization},
    )
    if status == 404:
        print("internal registry: tag already absent")
        return
    if not 200 <= status < 300:
        raise RuntimeError(f"internal registry: tag lookup failed with HTTP {status}")

    digest = headers.get("Docker-Content-Digest")
    if not digest:
        raise RuntimeError("internal registry: tag lookup did not return a digest")

    delete_url = (
        f"https://{resolved_host}/v2/{repository}/manifests/"
        f"{urllib.parse.quote(digest, safe=':')}"
    )
    status, _, _ = http_request(
        delete_url,
        method="DELETE",
        headers={"Accept": INTERNAL_ACCEPT, "Authorization": authorization},
    )
    if status in (202, 204):
        print("internal registry: tag deleted")
        return
    if status == 404:
        print("internal registry: tag already absent")
        return
    raise RuntimeError(f"internal registry: tag deletion failed with HTTP {status}")


def delete_dockerhub_tag(
    *,
    image: str,
    username: str,
    token: str,
) -> None:
    _, repository, tag = split_image_reference(image)
    repository = normalize_dockerhub_repository(repository)

    login_payload = json.dumps(
        {"username": username, "password": token},
    ).encode("utf-8")
    status, _, body = http_request(
        "https://hub.docker.com/v2/users/login/",
        method="POST",
        headers={"Content-Type": "application/json"},
        data=login_payload,
    )
    if status != 200:
        raise RuntimeError(f"docker hub: login failed with HTTP {status}")
    try:
        jwt = json.loads(body)["token"]
    except (KeyError, json.JSONDecodeError) as error:
        raise RuntimeError("docker hub: login response did not contain a token") from error

    delete_url = (
        f"https://hub.docker.com/v2/repositories/{repository}/tags/"
        f"{urllib.parse.quote(tag, safe='')}/"
    )
    status, _, _ = http_request(
        delete_url,
        method="DELETE",
        headers={"Authorization": f"Bearer {jwt}"},
    )
    if status in (200, 202, 204):
        print("docker hub: tag deleted")
        return
    if status == 404:
        print("docker hub: tag already absent")
        return
    raise RuntimeError(f"docker hub: tag deletion failed with HTTP {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", choices=("internal", "dockerhub"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--host", help="Internal registry host override")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.registry == "internal":
        username = os.environ.get("IREGISTRY_USERNAME", "")
        password = os.environ.get("IREGISTRY_PASSWORD", "")
        if not username or not password or not args.host:
            raise RuntimeError("internal registry cleanup requires host and credentials")
        delete_internal_registry_tag(
            host=args.host,
            image=args.image,
            username=username,
            password=password,
        )
    else:
        username = os.environ.get("DOCKERHUB_USERNAME", "")
        token = os.environ.get("DOCKERHUB_TOKEN", "")
        if not username or not token:
            raise RuntimeError("docker hub cleanup requires credentials")
        delete_dockerhub_tag(
            image=args.image,
            username=username,
            token=token,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as error:
        print(f"cleanup failed: {error}", file=sys.stderr)
        raise SystemExit(1)
