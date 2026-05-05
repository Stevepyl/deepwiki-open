import RepoInfo from "../types/repoinfo";

export default function getRepoUrl(repoInfo: RepoInfo): string {
  if (repoInfo.type === 'local' && repoInfo.localPath) {
    return repoInfo.localPath;
  }
  if (repoInfo.repoUrl) {
    return repoInfo.repoUrl;
  }
  if (repoInfo.owner && repoInfo.repo) {
    const host = repoInfo.type === "gitlab" ? "gitlab.com" : repoInfo.type === "bitbucket" ? "bitbucket.org" : "github.com";
    return `https://${host}/${repoInfo.owner}/${repoInfo.repo}`;
  }
  return '';
};
