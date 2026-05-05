import { NextRequest, NextResponse } from "next/server";

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || "http://localhost:8001";

export async function POST(req: NextRequest) {
  const requestBody = await req.json();
  const backendResponse = await fetch(`${TARGET_SERVER_BASE_URL}/chat/completions/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(requestBody),
  });

  if (!backendResponse.ok || !backendResponse.body) {
    return new NextResponse(await backendResponse.text(), {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
    });
  }

  const headers = new Headers();
  const contentType = backendResponse.headers.get("Content-Type");
  if (contentType) headers.set("Content-Type", contentType);
  headers.set("Cache-Control", "no-cache, no-transform");

  return new NextResponse(backendResponse.body, {
    status: backendResponse.status,
    headers,
  });
}
