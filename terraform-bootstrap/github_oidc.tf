variable "github_repo" {
  description = "GitHub repository in format: YOUR_USERNAME/YOUR_REPO"
  type        = string
}

variable "github_owner_id" {
  description = "Immutable GitHub owner (user/org) ID. Visible in the OIDC token sub claim: repo:OWNER@<id>/REPO@<id>:..."
  type        = string
  default     = ""
}

variable "github_repo_id" {
  description = "Immutable GitHub repository ID. Visible in the OIDC token sub claim: repo:OWNER@<id>/REPO@<id>:..."
  type        = string
  default     = ""
}

locals {
  github_repo_parts = split("/", var.github_repo)

  # GitHub changed the OIDC sub claim format: newer tokens carry immutable
  # numeric IDs (repo:OWNER@OWNER_ID/REPO@REPO_ID:*). Allow both formats.
  github_sub_patterns = var.github_owner_id != "" && var.github_repo_id != "" ? [
    "repo:${var.github_repo}:*",
    "repo:${local.github_repo_parts[0]}@${var.github_owner_id}/${local.github_repo_parts[1]}@${var.github_repo_id}:*",
  ] : ["repo:${var.github_repo}:*"]
}

data "tls_certificate" "github" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = data.tls_certificate.github.url
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github.certificates[0].sha1_fingerprint]
}

data "aws_iam_policy_document" "github_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = local.github_sub_patterns
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "github-actions-parking-pipeline-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_assume_role.json
}

resource "aws_iam_role_policy_attachment" "github_actions_deploy_policy" {
  role       = aws_iam_role.github_actions_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/AdministratorAccess"
}

output "github_actions_role_arn" {
  value = aws_iam_role.github_actions_deploy.arn
}
