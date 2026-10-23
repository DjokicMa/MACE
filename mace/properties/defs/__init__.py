"""Derived-property definitions, one module per area. Every module here is
imported by mace.properties.registry.load(); a new module registers its
properties with ``@derived(...)`` and needs no other edit. Modules whose name
starts with ``_`` are not loaded (helpers)."""
