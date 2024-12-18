# PSF Averaging Workflow

Create effective PSFs for all SPHEREx fiducial channels used to perform photometry. Assume that hundreds of exposures in the deep field NEP region smooth out PSFs azimuthally very well so that our effective, symmetrical PSFs are a good approximation. 

---
__Workflow__:

```mermaid
graph TD
    A[Start] --> B{Is it working?}
    B -->|Yes| C[Great!]
    B -->|No| D[Fix the issue]
    D --> A

