import { NextResponse } from "next/server";

const TARGET_SERVER_BASE_URL = process.env.SERVER_BASE_URL || "http://localhost:8001";

export async function GET() {
  try {
    const backendResponse = await fetch(`${TARGET_SERVER_BASE_URL}/models/config`, {
      method: "GET",
      headers: { Accept: "application/json" },
    });

    if (!backendResponse.ok) {
      return NextResponse.json(
        { error: `Backend service responded with status: ${backendResponse.status}` },
        { status: backendResponse.status },
      );
    }

    return NextResponse.json(await backendResponse.json());
  } catch (error) {
    const message = error instanceof Error ? error.message : "Error fetching model configurations";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
