variable "tenant_id" {
  type        = string
  description = "The ID of your tenant, provided by the Nebius AI team"
}

variable "project_id" {
  type        = string
  description = "The ID of your project, provided by the Nebius AI team"
}

variable "vm_username" {
  type        = string
  description = "The name of the user that will be created on virtual machines"
}

variable "vm_ssh_public_key_path" {
  type        = string
  description = "The path to a public key for SSH connections to virtual machines"
  default     = "~/.ssh/id_ed25519.pub"
}

variable "training_epochs" {
  type        = number
  description = "Number of epochs to train"
  default     = 10
}

variable "save_frequency" {
  type        = number
  description = "How often to save checkpoints"
  default     = 5
}

variable "training_batch_size" {
  type        = number
  description = "Batch size for training"
  default     = 64
}
