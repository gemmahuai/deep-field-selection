# PSF Averaging Workflow

Create effective PSFs for all SPHEREx fiducial channels used to perform photometry. Assume that hundreds of exposures in the deep field NEP region smooth out PSFs azimuthally very well so that our effective, symmetrical PSFs are a good approximation. 

---
__Workflow__:


```mermaid
 flowchart TB;

    %% Define input PSFs on the detector plane (mm)
    subgraph Row1[Input PSFs]
        direction LR
        PSF1(("PSF1<br>x, y (mm)"))
        PSF2(("PSF2<br>x, y (mm)"))
        PSF3(("PSF3<br>x, y (mm)"))
        Dots(("... ...")):::dot
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
        Dots_(("... ...")):::dot
        PSF52_(("PSF52<br>arr, ch No."))
        PSF53_(("PSF53<br>arr, ch No."))
        PSF54_(("PSF54<br>arr, ch No."))
        PSF1_~~~PSF2_~~~PSF3_~~~Dots_~~~PSF52_~~~PSF53_~~~PSF54_

        classDef dot fill:#fff,stroke:none;
    end

    Row1 -- Convert x,y (mm) to <br> array and channel number ---> Row2


    %% Define sorted PSFs with unique channels / arrays
    subgraph Row3[Sorted PSFs per unique channel]
        direction LR
        
        subgraph col1[ ]
        direction TB
        %% Define input PSFs in the same row
            CH1["Uniq CH 1<br>arr, ch No."]
            CH1 --> psfa(("PSF4"))
            CH1 --> psfb(("PSF29"))
            CH1 --> psfc(("PSF35"))
        end

        subgraph col2[ ]
        direction TB
        %% Define input PSFs in the same row
            CH2["Uniq CH 2<br>arr, ch No."]
            CH2 --> psfd(("PSF32"))
        end

        subgraph col3[ ]
        direction TB
        %% Define input PSFs in the same row
            CH3["Uniq CH n<br>arr, ch No."]
            CH3 --> psfe(("PSF1"))
            CH3 --> psff(("PSF2"))
        end
        Dots__["... ..."]:::dot

        col1 ~~~ col2 ~~~Dots__ ~~~ col3

        classDef dot fill:#fff,stroke:none;
    end

    Row2 -- Select unique channels ---> Row3

    %% Define averaged, asymmetrical PSF per unique channel
    subgraph Row4[Averaged, asymmetrical PSFs]
        direction LR
        
        PSFa1(("asym PSF,<br>Uniq CH 1"))
        PSFa2(("asym PSF,<br>Uniq CH 2"))
        PSFa3(("asym PSF,<br>Uniq CH n"))

        Dots___(("... ...")):::dot

        PSFa1 ~~~ PSFa2 ~~~ Dots___ ~~~ PSFa3
        
        classDef dot fill:#fff,stroke:none;
    end

    Row3 -- Average over all PSFs<br>for each unique channel ---> Row4


    %% Define rotationally averaged, symmetrical PSF per unique channel
    subgraph Row5[Azimuthally averaged, symmetrical PSFs]
        direction LR
        
        PSFs1(("sym PSF,<br>Uniq CH 1"))
        PSFs2(("sym PSF,<br>Uniq CH 2"))
        PSFs3(("sym PSF,<br>Uniq CH n"))

        Dots____(("... ...")):::dot

        PSFs1 ~~~ PSFs2 ~~~ Dots____ ~~~ PSFs3
        
        classDef dot fill:#fff,stroke:none;
    end

    Row4 -- Rot and avg PSF azimuthally<br>for each unique channel ---> Row5


    %% Define interpolated PSFs
    subgraph Row6[Interpolated, symmetrical PSFs]
        direction LR
        PSFs1_(("interp PSF,<br> fid CH 1"))
        PSFs2_(("interp PSF,<br> fid CH 2"))
        PSFs3_(("interp PSF,<br> fid CH 3"))
        DOTS(("... ...")):::dot
        PSFs100(("interp PSF,<br> fid CH 100"))
        PSFs101(("interp PSF,<br> fid CH 101"))
        PSFs102(("interp PSF,<br> fid CH 102"))
        PSFs1_~~~PSFs2_~~~PSFs3_~~~DOTS~~~PSFs100~~~PSFs101~~~PSFs102

        classDef dot fill:#fff,stroke:none;
    end 

    Row5 -- 2D Interpolation to all SPHEREx fiducial channels ---> Row6



```


