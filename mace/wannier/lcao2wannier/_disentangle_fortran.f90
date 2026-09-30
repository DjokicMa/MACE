! Copyright (c) 2025 Computational Materials Science Team (lcao2wannier, author William Comaskey).
! MIT License, see LICENSE in this directory.
! Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
module dis_kinds
  integer, parameter :: R8 = selected_real_kind(15, 307)
end module dis_kinds

!-----------------------------------------------------------------------
! mod_disentangle : frozen-constrained Omega_I minimization with PER-STATE
! masks — the compiled twin of lcao2wannier.hybrid.disentangle_frozen.
!
! Same fixed point, same Gauss-Seidel ordering (each k sees neighbours
! already updated this sweep; pure Jacobi diverges for this iteration),
! so the compiled and Python paths must agree to solver tolerance. The
! Python path stays the reference implementation and the trip-wire.
!
! Why compiled: after the batched-BLAS reform the residual cost of the
! sweep is interpreter dispatch on the per-k eigenproblem of small
! (<= nw) matrices — millions of sub-millisecond calls — and Gauss-Seidel
! is sequential in k, so Python cannot thread it. This kernel removes the
! dispatch; colored-k threading is the natural follow-up (the neighbour
! graph must be coloured so same-colour k are mutually non-adjacent).
!
! Two deliberate departures from the older window-based mod_omega kernel:
!   * the (dis,dis) block is accumulated DIRECTLY via ZHERK rank-k
!     updates, never by forming the full (nb,nb) Z and slicing it. Forming
!     Z is what made Sc scale (nb=441, nw=294) cost ~100 s/sweep.
!   * ZHEEVD (divide and conquer), matching what NumPy calls, rather than
!     ZHEEV.
!-----------------------------------------------------------------------
module mod_disentangle
  use dis_kinds, only: R8
  implicit none
  private
  public :: disentangle_frozen, omega_invariant_masked
  complex(R8), parameter :: ONE = (1._R8, 0._R8), ZERO = (0._R8, 0._R8)

contains

  !-- Omega_I = (1/Nk) sum_{k,b} w_b ( Nw - ||V_k^H M^{(k,b)} V_{k+b}||_F^2 )
  function omega_invariant_masked(U, mmn, neigh, w, nb, nw, nk, nntot) &
       &   result(omega)
    integer,     intent(in) :: nb, nw, nk, nntot
    complex(R8), intent(in) :: U(nb, nw, nk), mmn(nb, nb, nntot, nk)
    integer,     intent(in) :: neigh(nntot, nk)
    real(R8),    intent(in) :: w(nntot)
    real(R8) :: omega
    complex(R8) :: tmp(nb, nw), Mt(nw, nw)
    integer  :: k, ib, kn
    omega = 0._R8
    do k = 1, nk
       do ib = 1, nntot
          kn = neigh(ib, k)
          call zgemm('N','N', nb, nw, nb, ONE, mmn(1,1,ib,k), nb, &
               &     U(1,1,kn), nb, ZERO, tmp, nb)
          call zgemm('C','N', nw, nw, nb, ONE, U(1,1,k), nb, tmp, nb, &
               &     ZERO, Mt, nw)
          omega = omega + w(ib) * (dble(nw) - sum(abs(Mt)**2))
       end do
    end do
    omega = omega / dble(nk)
  end function omega_invariant_masked

  !-- top-na left singular vectors of A(m x n) -> V(m x na) --------------
  subroutine svd_left(Amat, m, n, na, V)
    integer,     intent(in)  :: m, n, na
    complex(R8), intent(in)  :: Amat(m, n)
    complex(R8), intent(out) :: V(m, na)
    complex(R8) :: Ac(m, n), Uf(m, min(m,n)), VT(1,1), wq(1)
    complex(R8), allocatable :: work(:)
    real(R8) :: S(min(m,n)), rwork(5*min(m,n))
    integer  :: info, lwork
    Ac = Amat
    call zgesvd('S','N', m, n, Ac, m, S, Uf, m, VT, 1, wq, -1, rwork, info)
    lwork = nint(real(wq(1))); allocate(work(lwork))
    call zgesvd('S','N', m, n, Ac, m, S, Uf, m, VT, 1, work, lwork, rwork, info)
    V = Uf(:, 1:na)
    deallocate(work)
  end subroutine svd_left

  !-- Hermitian eigensolve, ascending, divide and conquer.
  !   Workspaces are supplied by the caller: they are queried ONCE at the
  !   largest block size and reused, because the query itself is a LAPACK
  !   call and this runs nk times per sweep.
  subroutine heevd_ws(evec, lde, n, ev, work, lwork, rwork, lrwork, &
       &              iwork, liwork, info)
    integer,     intent(in)    :: lde, n, lwork, lrwork, liwork
    complex(R8), intent(inout) :: evec(lde, *)
    real(R8),    intent(out)   :: ev(*)
    complex(R8), intent(inout) :: work(*)
    real(R8),    intent(inout) :: rwork(*)
    integer,     intent(inout) :: iwork(*)
    integer,     intent(out)   :: info
    call zheevd('V','U', n, evec, lde, ev, work, lwork, rwork, lrwork, &
         &      iwork, liwork, info)
  end subroutine heevd_ws

  !---------------------------------------------------------------------
  ! Main kernel. Masks are int32 (0/1) so the f2py interface stays flat;
  ! froz/dis index lists are rebuilt here exactly as np.where would order
  ! them (ascending band index).
  !
  ! info: 0 ok, 1 infeasible masks at some k, 2 frozen not subset of admit.
  !---------------------------------------------------------------------
  subroutine disentangle_frozen(mmn, A, neigh, w, admit, frozen, &
       &                        nb, nw, nk, nntot, niter, tol, rel_tol, &
       &                        check_every, mix, U, omega, nsweeps, info)
    integer,     intent(in)  :: nb, nw, nk, nntot, niter, check_every
    complex(R8), intent(in)  :: mmn(nb, nb, nntot, nk), A(nb, nw, nk)
    integer,     intent(in)  :: neigh(nntot, nk)
    real(R8),    intent(in)  :: w(nntot), tol, rel_tol, mix
    integer,     intent(in)  :: admit(nb, nk), frozen(nb, nk)
    complex(R8), intent(out) :: U(nb, nw, nk)
    real(R8),    intent(out) :: omega
    integer,     intent(out) :: nsweeps, info

    integer, allocatable :: froz(:,:), dis(:,:), nfroz(:), ndis(:)
    complex(R8), allocatable :: MU(:,:), MUw(:,:), Zd(:,:), evec(:,:)
    complex(R8), allocatable :: Vt(:,:), Asub(:,:)
    complex(R8), allocatable :: zdprev(:,:,:), mmn_dis(:,:,:,:)
    real(R8), allocatable    :: ev(:)
    complex(R8), allocatable :: work(:)
    real(R8),    allocatable :: rwork(:)
    integer,     allocatable :: iwork(:)
    complex(R8) :: wq(1)
    real(R8)    :: rwq(1)
    integer     :: iwq(1)
    integer  :: k, ib, kn, m, i, j, c, nadd, it, ndmax, nd, ierr
    integer  :: lwork, lrwork, liwork
    real(R8) :: om, omp, delta
    logical  :: converged, do_mix
    logical, allocatable :: seeded(:)

    info = 0; omega = 0._R8; nsweeps = 0
    do_mix = (mix /= 1.0_R8)

    ! ---- masks -> index lists (ascending band index, as np.where) ------
    allocate(froz(nb,nk), dis(nb,nk), nfroz(nk), ndis(nk))
    do k = 1, nk
       nfroz(k) = 0; ndis(k) = 0
       do m = 1, nb
          if (frozen(m,k) /= 0) then
             if (admit(m,k) == 0) then; info = 2; return; end if
             nfroz(k) = nfroz(k) + 1; froz(nfroz(k),k) = m
          else if (admit(m,k) /= 0) then
             ndis(k) = ndis(k) + 1;  dis(ndis(k),k) = m
          end if
       end do
       if (nfroz(k) > nw .or. nfroz(k) + ndis(k) < nw) then
          info = 1; return
       end if
    end do
    ndmax = maxval(ndis)

    ! ---- initial subspace: frozen unit columns + SVD seed on dis rows --
    U = ZERO
    do k = 1, nk
       c = nfroz(k)
       do i = 1, c
          U(froz(i,k), i, k) = ONE
       end do
       nadd = nw - c
       if (nadd > 0) then
          nd = ndis(k)
          allocate(Asub(nd, nw), Vt(nd, nadd))
          do i = 1, nd
             Asub(i,:) = A(dis(i,k), :, k)
          end do
          call svd_left(Asub, nd, nw, nadd, Vt)
          do j = 1, nadd
             do i = 1, nd
                U(dis(i,k), c + j, k) = Vt(i,j)
             end do
          end do
          deallocate(Asub, Vt)
       end if
    end do

    if (do_mix) then
       allocate(zdprev(ndmax, ndmax, nk), seeded(nk))
       zdprev = ZERO; seeded = .false.
    end if

    ! ---- hoist the dis-row gather OUT of the sweep loop -----------------
    ! The masks never change during disentanglement, so the scattered dis
    ! rows of every M(k,b) are gathered ONCE into a contiguous buffer. Doing
    ! this per sweep (nk*nntot*niter gathers) is what made the first version
    ! of this kernel slower than NumPy, which hoists the same thing.
    allocate(mmn_dis(ndmax, nb, nntot, nk), stat=ierr)
    if (ierr /= 0) then; info = 3; return; end if
    do k = 1, nk
       do ib = 1, nntot
          do i = 1, ndis(k)
             mmn_dis(i, :, ib, k) = mmn(dis(i,k), :, ib, k)
          end do
       end do
    end do

    ! ---- one LAPACK workspace query at the largest block ---------------
    allocate(evec(ndmax, ndmax), Zd(ndmax, ndmax), ev(ndmax))
    call zheevd('V','U', ndmax, evec, ndmax, ev, wq, -1, rwq, -1, iwq, -1, &
         &      ierr)
    lwork = nint(real(wq(1))); lrwork = nint(rwq(1)); liwork = iwq(1)
    allocate(work(max(1,lwork)), rwork(max(1,lrwork)), iwork(max(1,liwork)))

    ! ---- Gauss-Seidel sweeps -------------------------------------------
    omp = huge(1._R8)
    converged = .false.
    allocate(MU(ndmax, nntot*nw), MUw(ndmax, nntot*nw))
    do it = 1, niter
       nsweeps = it
       do k = 1, nk
          nadd = nw - nfroz(k)
          if (nadd == 0) cycle
          nd = ndis(k)
          ! Accumulate over ALL neighbours in ONE BLAS-3 call. Issuing
          ! nntot separate rank-k updates is measurably slower than a
          ! single wide GEMM (the NumPy path does exactly this: it builds
          ! B = [MU_1 ... MU_nntot] and forms Zd = Bw B^H in one shot).
          do ib = 1, nntot
             kn = neigh(ib, k)
             c = (ib - 1) * nw
             call zgemm('N','N', nd, nw, nb, ONE, mmn_dis(1,1,ib,k), ndmax, &
                  &     U(1,1,kn), nb, ZERO, MU(1, c+1), ndmax)
             MUw(1:nd, c+1:c+nw) = w(ib) * MU(1:nd, c+1:c+nw)
          end do
          call zgemm('N','C', nd, nd, nntot*nw, ONE, MUw, ndmax, &
               &     MU, ndmax, ZERO, Zd, ndmax)
          if (do_mix) then
             if (seeded(k)) then
                Zd(1:nd,1:nd) = mix * Zd(1:nd,1:nd) &
                     &        + (1._R8 - mix) * zdprev(1:nd, 1:nd, k)
             end if
             zdprev(1:nd, 1:nd, k) = Zd(1:nd, 1:nd)
             seeded(k) = .true.
          end if
          evec(1:nd, 1:nd) = Zd(1:nd, 1:nd)
          call heevd_ws(evec, ndmax, nd, ev, work, lwork, rwork, lrwork, &
               &        iwork, liwork, ierr)          ! ascending
          U(:,:,k) = ZERO
          do i = 1, nfroz(k)
             U(froz(i,k), i, k) = ONE
          end do
          c = nfroz(k)
          do j = 0, nadd - 1                        ! top nadd, descending
             c = c + 1
             do i = 1, nd
                U(dis(i,k), c, k) = evec(i, nd - j)
             end do
          end do
       end do

       ! strided convergence monitor: comparing consecutive CHECKS spans
       ! check_every sweeps, so the test is strictly stronger, never weaker
       if (mod(it - 1, check_every) == 0 .or. it == niter) then
          om = omega_invariant_masked(U, mmn, neigh, w, nb, nw, nk, nntot)
          omega = om
          delta = abs(omp - om)
          if (delta < tol .or. delta < rel_tol * max(abs(om), 1._R8)) then
             converged = .true.
             exit
          end if
          omp = om
       end if
    end do
    if (.not. converged) omega = omega_invariant_masked(U, mmn, neigh, w, &
         &                                              nb, nw, nk, nntot)

    deallocate(MU, MUw, froz, dis, nfroz, ndis, mmn_dis)
    deallocate(evec, Zd, ev, work, rwork, iwork)
    if (do_mix) deallocate(zdprev, seeded)
  end subroutine disentangle_frozen

end module mod_disentangle
