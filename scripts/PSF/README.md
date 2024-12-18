# PSF Averaging Workflow

Create effective PSFs for all SPHEREx fiducial channels used to perform photometry. Assume that hundreds of exposures in the deep field NEP region smooth out PSFs azimuthally very well so that our effective, symmetrical PSFs are a good approximation. 

---
__Workflow__:

```mermaid
graph LR;
    %% Define input PSFs in the same row
    PSF1["PSF1<br>x, y (mm)"]:::circle
    PSF2["PSF2<br>x, y (mm)"]:::circle
    PSF3["PSF3<br>x, y (mm)"]:::circle
    PSF4["PSF4<br>x, y (mm)"]:::circle
    PSF5["PSF5<br>x, y (mm)"]:::circle
    Dots["... ..."]:::dot
    PSF52["PSF52<br>x, y (mm)"]:::circle
    PSF53["PSF53<br>x, y (mm)"]:::circle
    PSF54["PSF54<br>x, y (mm)"]:::circle

    %% Styling circles and dots
    classDef circle fill:#f9f,stroke:#333,stroke-width:2px;
    classDef dot fill:#fff,stroke:none;


```
