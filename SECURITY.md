# Security policy

## Supported versions

Security fixes are provided for the latest minor release of AutoOptLib.

## Loading algorithm files

Use the versioned JSON algorithm format for exchange and archival. Legacy
pickle files can execute arbitrary Python code when loaded; AutoOptLib emits a
warning, and users must load them only from trusted sources. Objective
functions are ordinary user-provided Python code and should be reviewed before
execution.

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue. Email
zhaoq@sustech.edu.cn with a description, reproduction steps, affected versions,
and any suggested mitigation. We will acknowledge the report and coordinate a
responsible disclosure timeline.
