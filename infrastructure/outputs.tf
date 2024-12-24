output "training_public_ip" {
  description = "The public IP address of the training VM"
  value = trimsuffix(nebius_compute_v1_instance.training_h100.status.network_interfaces[0].public_ip_address.address, "/32")
}
