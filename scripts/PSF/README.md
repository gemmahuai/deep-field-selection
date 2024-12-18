# PSF Averaging Workflow

Create effective PSFs for all SPHEREx fiducial channels used to perform photometry. Assume that hundreds of exposures in the deep field NEP region smooth out PSFs azimuthally very well so that our effective, symmetrical PSFs are a good approximation. 

---
__Workflow__:


```mermaid
 graph TD;

    %% Define input PSFs on the detector plane (mm)
    subgraph Row1[Input PSFs]
        direction LR
        PSF1(("PSF1<br>x, y (mm)"))
        PSF2(("PSF2<br>x, y (mm)"))
        PSF3(("PSF3<br>x, y (mm)"))
        Dots["... ..."]:::dot
        PSF52(("PSF52<br>x, y (mm)"))
        PSF53(("PSF53<br>x, y (mm)"))
        PSF54(("PSF54<br>x, y (mm)"))
        PSF1~~~PSF2~~~PSF3~~~Dots~~~PSF52~~~PSF53~~~PSF54

        classDef dot fill:#fff,stroke:none;
    end

    %% Define input PSFs converted to array and channel number
    subgraph Row2[Input PSFs]
        direction LR
        %% Define input PSFs in the same row
        PSF1_(("PSF1<br>arr, ch No."))
        PSF2_(("PSF2<br>arr, ch No."))
        PSF3_(("PSF3<br>arr, ch No."))
        Dots_["... ..."]:::dot
        PSF52_(("PSF52<br>arr, ch No."))
        PSF53_(("PSF53<br>arr, ch No."))
        PSF54_(("PSF54<br>arr, ch No."))
        PSF1_~~~PSF2_~~~PSF3_~~~Dots_~~~PSF52_~~~PSF53_~~~PSF54_

        classDef dot fill:#fff,stroke:none;
    end

    Row1 -- Convert x,y (mm) to <br> array and channel number --> Row2


    %% Define sorted PSFs with unique channels / arrays
    subgraph Row3[Sorted PSFs per unique channel]
        direction LR
        %% Define input PSFs in the same row
        CH1(("PSF1<br>arr, ch No."))
        CH2(("PSF2<br>arr, ch No."))
        Dots__["... ..."]:::dot
        CH3(("PSF2<br>arr, ch No."))
        CH4(("PSF52<br>arr, ch No."))
        CH1~~~CH2~~~Dots__~~~CH3~~~CH4

        classDef dot fill:#fff,stroke:none;
    end

    Row2 -- Select unique channels --> Row3






```


