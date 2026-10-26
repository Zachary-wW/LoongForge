# Vendored CSA utilities

These modules are a LoongForge-owned compatibility copy of the packed THD CSA
utilities developed in Loong-Megatron commit `6605602ea` ("Add packed THD
context parallelism for DSv4"). The feature is newer than the v0.19.2 baseline
and cannot be imported from a stock `core_v0.19.2` checkout.

Keep this package until Loong-Megatron publishes an integration branch that
contains the packed CSA implementation. At that point compare the vendored
files with upstream and replace them with imports where semantics match.
