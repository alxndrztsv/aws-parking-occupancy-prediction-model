resource "aws_ssm_parameter" "api_login" {
  name      = "/${local.prefix}/api-login"
  type      = "SecureString"
  value     = var.api_login
  overwrite = true
}

resource "aws_ssm_parameter" "api_password" {
  name      = "/${local.prefix}/api-password"
  type      = "SecureString"
  value     = var.api_password
  overwrite = true
}
