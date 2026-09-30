! Copyright (c) 2025 William Comaskey. MIT License, see LICENSE in this directory.
! Vendored into MACE from lcao2wannier 1.0.0; local changes are listed in VENDORED.md.
module spread_kinds
  integer, parameter :: R8 = selected_real_kind(15, 307)
end module spread_kinds

!-----------------------------------------------------------------------
! mod_omega : gauge-invariant spread Omega_I from precomputed overlaps.
!
! Omega_I = (1/Nk) sum_{k,b} w_b ( Nw - || V_k^H M^{(k,b)} V_{k+b} ||_F^2 )
!
! where M^{(k,b)}_{mn} = <u_mk|u_n,k+b> are the .mmn overlaps, V_k is an
! orthonormal Nb x Nw basis of the selected subspace, and w_b are the
! Marzari-Vanderbilt finite-difference weights.  Depends only on the
! SUBSPACE (gauge-invariant) -> evaluable without maximal localization,
! and cheap to re-evaluate for many candidate windows once M and the
! projections exist.
!-----------------------------------------------------------------------
module mod_omega
  use spread_kinds, only: R8
  implicit none
  private
  public :: mv_weights, svd_orthonormal, omega_invariant, disentangle_omega
contains

  !-- Marzari-Vanderbilt weights: solve sum_b w_b b_a b_b = delta_ab ----
  subroutine mv_weights(bcart, nb, w, max_resid)
    integer,  intent(in)  :: nb
    real(R8), intent(in)  :: bcart(3, nb)   ! Cartesian b-vectors (1/Ang)
    real(R8), intent(out) :: w(nb)
    real(R8), intent(out) :: max_resid
    integer  :: i, j, s, ns, shell(nb)
    real(R8) :: bn(nb), shval(nb), comp(3,3), d
    real(R8), allocatable :: Mat(:,:), wshell(:)
    real(R8), parameter   :: TOL = 1.0e-6_R8

    do i = 1, nb
       bn(i) = sqrt(sum(bcart(:,i)**2))
    end do
    ! group into shells of equal |b|
    ns = 0
    do i = 1, nb
       s = 0
       do j = 1, ns
          if (abs(bn(i) - shval(j)) < TOL) then; s = j; exit; end if
       end do
       if (s == 0) then; ns = ns + 1; shval(ns) = bn(i); s = ns; end if
       shell(i) = s
    end do
    ! 6 x ns matrix of shell second moments
    allocate(Mat(6, ns), wshell(ns)); Mat = 0
    do i = 1, nb
       s = shell(i)
       Mat(1,s) = Mat(1,s) + bcart(1,i)*bcart(1,i)
       Mat(2,s) = Mat(2,s) + bcart(2,i)*bcart(2,i)
       Mat(3,s) = Mat(3,s) + bcart(3,i)*bcart(3,i)
       Mat(4,s) = Mat(4,s) + bcart(1,i)*bcart(2,i)
       Mat(5,s) = Mat(5,s) + bcart(2,i)*bcart(3,i)
       Mat(6,s) = Mat(6,s) + bcart(3,i)*bcart(1,i)
    end do
    call solve_lstsq(Mat, (/1._R8,1._R8,1._R8,0._R8,0._R8,0._R8/), 6, ns, wshell)
    do i = 1, nb
       w(i) = wshell(shell(i))
    end do
    ! completeness residual
    comp = 0
    do i = 1, nb
       do j = 1, 3
          comp(j,1) = comp(j,1) + w(i)*bcart(j,i)*bcart(1,i)
          comp(j,2) = comp(j,2) + w(i)*bcart(j,i)*bcart(2,i)
          comp(j,3) = comp(j,3) + w(i)*bcart(j,i)*bcart(3,i)
       end do
    end do
    max_resid = 0
    do i = 1, 3
       do j = 1, 3
          d = comp(i,j); if (i == j) d = d - 1._R8
          max_resid = max(max_resid, abs(d))
       end do
    end do
    deallocate(Mat, wshell)
  end subroutine mv_weights

  !-- least-squares solve A x = b (A: m x n, m>=n) via LAPACK dgels -----
  subroutine solve_lstsq(A, b, m, n, x)
    integer,  intent(in)    :: m, n
    real(R8), intent(inout) :: A(m, n)
    real(R8), intent(in)    :: b(m)
    real(R8), intent(out)   :: x(n)
    real(R8) :: bb(max(m,n)), wq(1)
    real(R8), allocatable :: work(:)
    integer :: info, lwork
    bb = 0; bb(1:m) = b
    call dgels('N', m, n, 1, A, m, bb, max(m,n), wq, -1, info)
    lwork = nint(wq(1)); allocate(work(lwork))
    call dgels('N', m, n, 1, A, m, bb, max(m,n), work, lwork, info)
    x = bb(1:n); deallocate(work)
  end subroutine solve_lstsq

  !-- V (nb x nw) = left singular vectors of A (orthonormal subspace) ---
  subroutine svd_orthonormal(A, nb, nw, V)
    integer,     intent(in)  :: nb, nw
    complex(R8), intent(in)  :: A(nb, nw)
    complex(R8), intent(out) :: V(nb, nw)
    complex(R8) :: Acopy(nb, nw), VT(1,1), wq(1)
    complex(R8), allocatable :: work(:)
    real(R8)    :: S(nw), rwork(5*nw)
    integer     :: info, lwork
    Acopy = A
    call zgesvd('S','N', nb, nw, Acopy, nb, S, V, nb, VT, 1, wq, -1, rwork, info)
    lwork = nint(real(wq(1))); allocate(work(lwork))
    call zgesvd('S','N', nb, nw, Acopy, nb, S, V, nb, VT, 1, work, lwork, rwork, info)
    deallocate(work)
  end subroutine svd_orthonormal

  !-- Omega_I trace over precomputed overlaps + subspaces --------------
  function omega_invariant(V, mmn, neigh, w, nb, nw, nk, nntot) result(omega)
    integer,     intent(in) :: nb, nw, nk, nntot
    complex(R8), intent(in) :: V(nb, nw, nk)
    complex(R8), intent(in) :: mmn(nb, nb, nntot, nk)
    integer,     intent(in) :: neigh(nntot, nk)
    real(R8),    intent(in) :: w(nntot)
    real(R8) :: omega
    complex(R8) :: tmp(nb, nw), Mt(nw, nw)
    complex(R8), parameter :: ONE=(1._R8,0._R8), ZERO=(0._R8,0._R8)
    integer  :: k, ib, kn
    omega = 0
    do k = 1, nk
       do ib = 1, nntot
          kn = neigh(ib, k)
          call zgemm('N','N', nb, nw, nb, ONE, mmn(1,1,ib,k), nb, &
               &     V(1,1,kn), nb, ZERO, tmp, nb)
          call zgemm('C','N', nw, nw, nb, ONE, V(1,1,k), nb, tmp, nb, &
               &     ZERO, Mt, nw)
          omega = omega + w(ib) * (dble(nw) - sum(abs(Mt)**2))
       end do
    end do
    omega = omega / dble(nk)
  end function omega_invariant

  !-- SMV disentanglement: minimized Omega_I for a given energy window ----
  !   Frozen bands (eig in [fmin,fmax]) are forced; the rest of the subspace
  !   is the projection-optimal complement, then iterated to minimize Omega_I.
  !   info=0 ok; info=1 invalid window (n_froz>nw or n_outer<nw at some k).
  subroutine disentangle_omega(mmn, A, eig, neigh, w, nb, nw, nk, nntot, &
       &                       fmin, fmax, wmin, wmax, niter, tol, omega, info)
    integer,     intent(in)  :: nb, nw, nk, nntot, niter
    complex(R8), intent(in)  :: mmn(nb,nb,nntot,nk), A(nb,nw,nk)
    real(R8),    intent(in)  :: eig(nb,nk), w(nntot), fmin, fmax, wmin, wmax, tol
    integer,     intent(in)  :: neigh(nntot,nk)
    real(R8),    intent(out) :: omega
    integer,     intent(out) :: info
    complex(R8), allocatable :: U(:,:,:), MU(:,:), Z(:,:), Zd(:,:), evec(:,:), Vt(:,:)
    integer,     allocatable :: froz(:,:), dis(:,:), nfroz(:), ndis(:)
    real(R8),    allocatable :: ev(:)
    integer :: k, ib, kn, m, i, j, c, nadd, it
    real(R8) :: om, omp
    complex(R8), parameter :: ONE=(1._R8,0._R8), ZERO=(0._R8,0._R8)

    info = 0
    allocate(froz(nb,nk), dis(nb,nk), nfroz(nk), ndis(nk))
    do k = 1, nk
       nfroz(k)=0; ndis(k)=0
       do m = 1, nb
          if (eig(m,k) >= fmin .and. eig(m,k) <= fmax) then
             nfroz(k)=nfroz(k)+1; froz(nfroz(k),k)=m
          else if (eig(m,k) >= wmin .and. eig(m,k) <= wmax) then
             ndis(k)=ndis(k)+1; dis(ndis(k),k)=m
          end if
       end do
       if (nfroz(k) > nw .or. nfroz(k)+ndis(k) < nw) then; info=1; return; end if
    end do

    allocate(U(nb,nw,nk))
    ! init: frozen unit columns + SVD of A on disentangle bands
    do k = 1, nk
       U(:,:,k) = ZERO; c=0
       do i = 1, nfroz(k); c=c+1; U(froz(i,k),c,k)=ONE; end do
       nadd = nw - nfroz(k)
       if (nadd > 0) then
          allocate(Zd(ndis(k),nw), Vt(ndis(k),nadd))
          do i=1,ndis(k); Zd(i,:)=A(dis(i,k),:,k); end do
          call svd_left(Zd, ndis(k), nw, nadd, Vt)
          do j=1,nadd; c=c+1; do i=1,ndis(k); U(dis(i,k),c,k)=Vt(i,j); end do; end do
          deallocate(Zd, Vt)
       end if
    end do

    omp = 1.0e30_R8
    allocate(MU(nb,nw), Z(nb,nb))
    do it = 1, niter
       do k = 1, nk
          nadd = nw - nfroz(k)
          if (nadd == 0) cycle
          Z = ZERO
          do ib = 1, nntot
             kn = neigh(ib,k)
             call zgemm('N','N', nb, nw, nb, ONE, mmn(1,1,ib,k), nb, U(1,1,kn), nb, ZERO, MU, nb)
             call zgemm('N','C', nb, nb, nw, cmplx(w(ib),0._R8,R8), MU, nb, MU, nb, ONE, Z, nb)
          end do
          allocate(Zd(ndis(k),ndis(k)), ev(ndis(k)), evec(ndis(k),ndis(k)))
          do i=1,ndis(k); do j=1,ndis(k); Zd(i,j)=Z(dis(i,k),dis(j,k)); end do; end do
          call heev_all(Zd, ndis(k), ev, evec)            ! ascending eigvals
          U(:,:,k)=ZERO; c=0
          do i=1,nfroz(k); c=c+1; U(froz(i,k),c,k)=ONE; end do
          do j=ndis(k), ndis(k)-nadd+1, -1                ! top nadd
             c=c+1; do i=1,ndis(k); U(dis(i,k),c,k)=evec(i,j); end do
          end do
          deallocate(Zd, ev, evec)
       end do
       om = omega_invariant(U, mmn, neigh, w, nb, nw, nk, nntot)
       if (abs(omp-om) < tol) exit
       omp = om
    end do
    omega = om
    deallocate(U, MU, Z, froz, dis, nfroz, ndis)
  end subroutine disentangle_omega

  !-- top-na left singular vectors of A(m x n) into V(m x na) -----------
  subroutine svd_left(Amat, m, n, na, V)
    integer,     intent(in)  :: m, n, na
    complex(R8), intent(in)  :: Amat(m, n)
    complex(R8), intent(out) :: V(m, na)
    complex(R8) :: Ac(m,n), Uf(m,min(m,n)), VT(1,1), wq(1)
    complex(R8), allocatable :: work(:)
    real(R8) :: S(min(m,n)), rwork(5*min(m,n))
    integer  :: info, lwork
    Ac = Amat
    call zgesvd('S','N', m, n, Ac, m, S, Uf, m, VT, 1, wq, -1, rwork, info)
    lwork=nint(real(wq(1))); allocate(work(lwork))
    call zgesvd('S','N', m, n, Ac, m, S, Uf, m, VT, 1, work, lwork, rwork, info)
    V = Uf(:,1:na); deallocate(work)
  end subroutine svd_left

  !-- full Hermitian eigensolve (ascending) ----------------------------
  subroutine heev_all(Hmat, n, ev, evec)
    integer,     intent(in)  :: n
    complex(R8), intent(in)  :: Hmat(n,n)
    real(R8),    intent(out) :: ev(n)
    complex(R8), intent(out) :: evec(n,n)
    complex(R8) :: wq(1)
    complex(R8), allocatable :: work(:)
    real(R8) :: rwork(max(1,3*n-2))
    integer  :: info, lwork
    evec = Hmat
    call zheev('V','U', n, evec, n, ev, wq, -1, rwork, info)
    lwork=nint(real(wq(1))); allocate(work(lwork))
    call zheev('V','U', n, evec, n, ev, work, lwork, rwork, info)
    deallocate(work)
  end subroutine heev_all

end module mod_omega
