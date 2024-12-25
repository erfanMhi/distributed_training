locals {
  fs_device_name = "training-vm-filesystem-1"
}

data "nebius_vpc_v1_subnet" "default" {
  parent_id = var.project_id
  name      = "default-subnet-jvzrrihy"
}

resource "nebius_compute_v1_disk" "training_boot_200gib" {
  parent_id           = var.project_id
  type                = "NETWORK_SSD"
  block_size_bytes    = 4096
  name                = "training-vm-disk-1"
  size_bytes          = 214748364800 # 200 GiB
  source_image_family = {
    image_family = "ubuntu22.04-cuda12"
  }
}

resource "nebius_compute_v1_filesystem" "training_1024gib" {
  parent_id        = var.project_id
  name             = "training-vm-filesystem-1"
  type             = "NETWORK_SSD"
  block_size_bytes = 4096
  size_bytes       = 107374182400  # 100 GiB (100 * 1024 * 1024 * 1024 bytes)
}

resource "nebius_compute_v1_gpu_cluster" "training_fabric_3" {
  parent_id         = var.project_id
  name              = "training-demo-cluster"
  infiniband_fabric = "fabric-3"
}

resource "nebius_compute_v1_instance" "training_h100" {
  name = "training-h100"
  parent_id = var.project_id
  gpu_cluster = {
    id = nebius_compute_v1_gpu_cluster.training_fabric_3.id
  }
  network_interfaces = [
    {
      name              = "eth0"
      subnet_id         = data.nebius_vpc_v1_subnet.default.id
      ip_address        = {}
      public_ip_address = {}
    }
  ]
  resources = {
    platform = "gpu-h100-sxm"
    preset   = "8gpu-128vcpu-1600gb"
  }
  boot_disk = {
    attach_mode   = "READ_WRITE"
    existing_disk = {
      id = nebius_compute_v1_disk.training_boot_200gib.id
    }
  }
  filesystems = [
    {
      attach_mode         = "READ_WRITE",
      mount_tag           = local.fs_device_name
      existing_filesystem = {
        id = nebius_compute_v1_filesystem.training_1024gib.id
      }
    }
  ]
  cloud_init_user_data = templatefile("${path.module}/scripts/cloud-init.tftpl", {
    vm_username     = var.vm_username
    ssh_public_key  = file(var.vm_ssh_public_key_path)
    fs_device_name  = local.fs_device_name
    epochs          = var.training_epochs
    save_frequency  = var.save_frequency
    batch_size      = var.training_batch_size
  })
}
