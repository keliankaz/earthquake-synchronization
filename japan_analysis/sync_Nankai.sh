selfdir=$(dirname $0)
N2=2048 # n points on grid, length = 81920 m , 2048 elements for Dc = 1 cm 
DX=40  # cell size m
Dc=0.01
Lratio=1.1 # L1/L2 = 1.1 ----- Nankai to Tonankai length ratio
# L2=5000  # Tonankai, E
# Sedge=$(echo "$L1 * 0.1 / 2" | bc -l)  # edge (absolute) of velocity-weakening barrier
#### parameters to change 
# Dc 
# S 
# fixed stuff for naming


for Sedge in 100 1000 3000 10000; do
    for L2 in 500 1000 3000 5000 10000; do
        #L1=$(echo "$L2 * $Lratio" | bc -l)  # Nankai
        L1=$(echo "$L2 * $Lratio" | bc -l)  # Nankai
        WDIR="$selfdir/Nankai_output/Dc_01/Dc_01_L1_${L1%.*}_m_L2_${L2%.*}_m_S_${Sedge%.*}_m"

        if [ ! -e $WDIR ]; then
            echo adding directory "$WDIR"
            mkdir "$WDIR"
        fi
        # use --verbose=2 to output all parameters
        # use --verbose=1 to output only parameters different from previous patch
        OMP_NUM_THREADS=70 /home/alba/Documents/motorcycle/2d/antiplane/build/motorcycle-ap-ratestate-serial \
            --verbose 1 \
            --epsilon 1e-6 \
            --export-state \
            --export-stress \
            --export-netcdf \
            --export-netcdf-rate 20 \
            --export-netcdf-step 4 \
            --maximum-step 3.15e7 \
            --maximum-iterations 5000000 \
            --friction-law 1 <<EOF
# output directory
$WDIR
# Rigidity
30e3 
# time interval
3.5e10      # KDC
# number of faults
1
# grid dimension (N2)
$N2
# sampling (dx2)
$DX
#   n  tau0   mu0   sig   a   b   L   Vo   G/(2Vs)   Vl Dirichlet Sedge L1 L2
$(echo "" | awk -v n2="$N2" -v dx="$DX" -v Dc="$Dc" -v Sedge="$Sedge" -v L1="$L1" -v L2="$L2" '
    function abs(x){return (x>0)?x:-x};
    BEGIN{
        c=1;
        tau0_p=-1; mu0_p=-1; sig_p=-1;
        a_p=-1; b_p=-1; L_p=-1;
        Vo_p=-1; damping_p=-1;
        Vl_p=-1; dirichlet_p="T";
    }{
        for (i2=0; i2<n2; i2++) {
            x2=(i2-n2/2)*dx; 
            tau0=-1;   # Initial shear traction (MPa); use -1 for steady-state
            L=Dc;      # Characteristic weakening distance (m)
            a=1e-2;    # Rate-dependent parameter (unitless)
            
            nankai_start = -Sedge - L1;
            nankai_end = -Sedge;
            tonankai_start = Sedge;
            tonankai_end = Sedge + L2;
            outer_limit = 40e3;

            if (x2 >= nankai_start && x2 <= nankai_end) {
                b = a + 4.0e-3;
            } else if (x2 >= tonankai_start && x2 <= tonankai_end) {
                b = a + 4.0e-3;
            } else if (abs(x2) < Sedge) {
                b = a - 4.0e-3;
            } else if (abs(x2) <= outer_limit) {
                b = a - 4.0e-3;
            } else {
                b = a - 4.0e-3;
            }

            mu0=0.6;   # Reference coefficient of friction (unitless)
            sig=1e2;   # Effective normal stress (MPa)
            Vo=1e-6;   # Reference slip-rate (m/s)
            Vl=1e-9;   # Loading rate (m/s)
            damping=5; # Radiation damping coefficient (MPa/m/s)
			# Dirichlet BC for 12 km on each outer loading zone
			dirichlet_cells = int(12e3 / dx);
			if (i2 < dirichlet_cells || i2 >= (n2 - dirichlet_cells)) {
				dirichlet = "T"; # Apply Dirichlet boundary condition
			} else {
				dirichlet = "F"; # Resolve friction law
			}
            # Check if all parameters are identical to the previous patch
            if ((tau0_p == tau0) && (mu0_p==mu0) && (sig_p==sig) && (a_p==a) && (b_p==b) && 
                (L_p==L) && (Vo_p==Vo) && (damping_p==damping) && (Vl_p==Vl) && 
                (dirichlet_p==dirichlet)) {
                # Print minus line number to save space (previous value is used)
                printf "%5d\n", -c;
            } else {
                # Print new set of parameters
                printf "%5d %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %10.2e %d %10.2e %s\n", 
                        c, tau0, mu0, sig, a, b, L, Vo, damping, Vl, dirichlet;
            }
            c++;
            tau0_p=tau0; mu0_p=mu0; sig_p=sig;
            a_p=a; b_p=b; L_p=L;
            Vo_p=Vo; damping_p=damping;
            Vl_p=Vl; dirichlet_p=dirichlet;
        }
    }')
# number of observation patches
3
# -----------------------------------------------------------------------------------
#   n fault     i2 rate
# -----------------------------------------------------------------------------------
    1     1 $(awk -v n2=$N2 'BEGIN{print int(n2/2)+1}') 1                     # Center of Fault 1
    2     1 $(awk -v n2=$N2 -v dx=$DX 'BEGIN{print int(n2/2)-(2400/dx)}') 1   # 2.4 km to the left of the center 
    3     1 $(awk -v n2=$N2 -v dx=$DX 'BEGIN{print int(n2/2)+(2400/dx)}') 1   # 2.4 km to the right of the center 
# number of events (not implemented)
0
EOF
done
done