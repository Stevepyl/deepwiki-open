import { NextResponse } from "next/server";

const BACKEND_URL = process.env.PYTHON_BACKEND_HOST || "http://localhost:8001";

export async function GET() {
  const response = await fetch(`${BACKEND_URL}/api/processed_projects`, {
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
  });
  const data = await response.json();
  return NextResponse.json(data, { status: response.status });
}

export async function DELETE(request: Request) {
  const body = (await request.json()) as {
    owner?: string;
    repo?: string;
    repo_type?: string;
    language?: string;
  };

  if (!body.owner || !body.repo || !body.repo_type || !body.language) {
    return NextResponse.json({ error: "owner, repo, repo_type, and language are required." }, { status: 400 });
  }

  const params = new URLSearchParams({
    owner: body.owner,
    repo: body.repo,
    repo_type: body.repo_type,
    language: body.language,
  });
  const response = await fetch(`${BACKEND_URL}/api/wiki_cache?${params}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
  });
  const data = await response.json();
  return NextResponse.json(data, { status: response.status });
}
