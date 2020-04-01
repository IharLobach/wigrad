from scipy.special import sinc, jv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class UnknownPolarizationTypeError(ValueError):

    def __init__(self):
        super().__init__("Unknown polarization type. "
                         "Choose from 'x', 'y' and 'sum'.")


class Wiggler():
    def __init__(self, K_peak=1, N_periods=10, lambda_wiggler_m=0.055):
        self.K_peak = K_peak
        self.N_periods = N_periods
        self.lambda_wiggler_m = lambda_wiggler_m
        self.aux_const = 1+self.K_peak**2/2


class WigglerRadiationSimulator():
    alpha = 1/137
    label_font_size = 20
    label_linespacing = 1

    def __init__(self,
                 wiggler,
                 mesh,
                 gamma=100/0.511,
                 harmonics=[1],
                 bessel_cutoff=10,
                 aperture=None,
                 only_calc_sum_of_both_polarizations=False,
                 spectral_transmission=None):
        self.wiggler = wiggler
        self.gamma = gamma
        self.harmonics = harmonics
        self.bessel_cutoff = bessel_cutoff
        if aperture not in [None, 'ellipse']:
            raise ValueError("Unknown aperture type. "
                             "Choose from None and 'ellipse'")
        self.aperture = aperture
        self.only_calc_sum_of_both_polarizations = \
            only_calc_sum_of_both_polarizations
        self.lambda1_um = 1e6*self.wiggler.lambda_wiggler_m\
            / 2/self.gamma**2*self.wiggler.aux_const
        self.x_range, self.y_range, self.lambda_range = mesh
        self.x_2D = np.tile(self.x_range, (self.n_y, 1))
        self.y_2D = np.tile(self.y_range.reshape(-1, 1), (1, self.n_x))
        self.x_step = (self.x_range[-1]-self.x_range[0])/(self.n_x-1)
        self.y_step = (self.y_range[-1]-self.y_range[0])/(self.n_y-1)
        self.lambda_step = (self.lambda_range[-1]-self.lambda_range[0])\
            / (self.n_lambda-1)
        if spectral_transmission is None:
            self.spectral_transmission = np.ones(self.n_lambda)
        elif len(spectral_transmission) != self.n_lambda:
            raise ValueError("Length of spectral_transmission must be"
                             " the same as length of wavelength array.")
        else:
            self.spectral_transmission = spectral_transmission

    @property
    def n_x(self):
        return len(self.x_range)

    @property
    def n_y(self):
        return len(self.y_range)

    @property
    def n_lambda(self):
        return len(self.lambda_range)

    @property
    def x_3D(self):
        return np.tile(self.x_2D, (self.n_lambda, 1, 1))

    @property
    def y_3D(self):
        return np.tile(self.y_2D, (self.n_lambda, 1, 1))

    def __calc_photon_flux_on_meshgrid_one_harmonic(self, harmonic):
        x_2D = self.x_2D
        y_2D = self.y_2D
        r2_2D = self.gamma**2*(x_2D**2+y_2D**2)
        A = self.wiggler.aux_const+r2_2D
        Y = harmonic*self.wiggler.K_peak**2/4/A
        X = 2*harmonic*self.gamma*self.wiggler.K_peak*x_2D/A
        sum1 = 0
        sum2 = 0
        sum3 = 0
        p = -self.bessel_cutoff
        jv2pm1 = jv(harmonic+2*p-1, X)
        for p in range(-self.bessel_cutoff, self.bessel_cutoff+1):
            jvpY = jv(p, Y)
            sum1 += jv(harmonic+2*p, X)*jvpY
            sum2 += jv2pm1*jvpY
            jv2pp1 = jv(harmonic+2*p+1, X)
            sum3 += jv2pp1*jvpY
            jv2pm1 = jv2pp1
        aux_factor = self.alpha*harmonic*self.gamma**2\
            * self.wiggler.N_periods**2\
            / A**2
        bessel_part_x = aux_factor \
            * np.absolute(2*self.gamma*x_2D*sum1
                          - self.wiggler.K_peak*(sum2+sum3))**2
        bessel_part_y = aux_factor*np.absolute(2*self.gamma*y_2D*sum1)**2
        dw_arr = self.lambda1_um/self.lambda_range-harmonic
        L = [(sinc(self.wiggler.N_periods*(harmonic*r2_2D+dw*A)
              / self.wiggler.aux_const))**2 / l * st for dw, l, st in
             zip(dw_arr, self.lambda_range, self.spectral_transmission)]
        L = np.asarray(L)
        if self.only_calc_sum_of_both_polarizations:
            return (bessel_part_x + bessel_part_y)*L
        else:
            return bessel_part_x*L, bessel_part_y*L

    def calc_photon_flux_on_meshgrid(self):
        res = \
            self.__calc_photon_flux_on_meshgrid_one_harmonic(self.harmonics[0])
        for h in self.harmonics[1:]:
            res = np.sum(
                (
                    res,
                    self.__calc_photon_flux_on_meshgrid_one_harmonic(h)
                ), axis=0)
        if self.only_calc_sum_of_both_polarizations:
            self.__photon_flux_3D_sum_both_polarizations = res
        else:
            self.__photon_flux_3D_polarization_x = res[0]
            self.__photon_flux_3D_polarization_y = res[1]
        del res
        if self.aperture == 'ellipse':
            x_max = max(self.x_range)
            y_max = max(self.y_range)
            elliptic_aperture = \
                (self.x_3D**2/x_max**2+self.y_3D**2/y_max**2) < 1
            if self.only_calc_sum_of_both_polarizations:
                self.__photon_flux_3D_sum_both_polarizations = \
                    np.where(elliptic_aperture,
                             self.__photon_flux_3D_sum_both_polarizations,
                             0)
            else:
                self.__photon_flux_3D_polarization_x = \
                    np.where(elliptic_aperture,
                             self.__photon_flux_3D_polarization_x,
                             0)
                self.__photon_flux_3D_polarization_y = \
                    np.where(elliptic_aperture,
                             self.__photon_flux_3D_polarization_y,
                             0)

    def __get_photon_flux_3D_sum_both_polarizations(self):
        if self.only_calc_sum_of_both_polarizations:
            return self.__photon_flux_3D_sum_both_polarizations
        else:
            return self.__photon_flux_3D_polarization_x \
                + self.__photon_flux_3D_polarization_y

    def get_photon_flux_3D(self,
                           polarization='sum'):
        if polarization == 'sum':
            return self.__get_photon_flux_3D_sum_both_polarizations()
        elif polarization == 'x':
            return self.__photon_flux_3D_polarization_x
        elif polarization == 'y':
            return self.__photon_flux_3D_polarization_y
        else:
            raise UnknownPolarizationTypeError()

    def set_photon_flux_3D(self,
                           polarization,
                           value):
        if polarization == 'sum':
            self.__photon_flux_3D_sum_both_polarizations = value
        elif polarization == 'x':
            self.__photon_flux_3D_polarization_x = value
        elif polarization == 'y':
            self.__photon_flux_3D_polarization_y = value
        else:
            raise UnknownPolarizationTypeError()

    def __extend_angular_mesh_using_symmetries(self):
        x_range1 = self.x_range
        x_range2 = np.flip(-x_range1)
        self.x_range = np.concatenate((x_range2, x_range1))
        y_range1 = self.y_range
        y_range2 = np.flip(-y_range1)
        self.y_range = np.concatenate((y_range2, y_range1))
        x1 = self.x_2D
        y1 = self.y_2D
        x2 = np.flip(-x1, axis=1)
        y2 = y1
        x21 = np.hstack((x2, x1))
        y21 = np.hstack((y2, y1))
        x34 = x21
        y34 = np.flip(-y21, axis=0)
        self.x_2D = np.vstack((x34, x21))
        self.y_2D = np.vstack((y34, y21))

    def __extend_photon_flux_using_symmetries(self, z):
        z1 = z
        z2 = np.flip(z1, axis=2)
        z21 = np.concatenate((z2, z1), axis=2)
        z34 = np.flip(z21, axis=1)
        return np.concatenate((z34, z21), axis=1)

    def extend_results_using_symmetries(self):
        self.__extend_angular_mesh_using_symmetries()
        if self.only_calc_sum_of_both_polarizations:
            self.__photon_flux_3D_sum_both_polarizations = \
                self.__extend_photon_flux_using_symmetries(
                    self.__get_photon_flux_3D_sum_both_polarizations())
        else:
            self.__photon_flux_3D_polarization_x = \
                self.__extend_photon_flux_using_symmetries(
                    self.__photon_flux_3D_polarization_x)
            self.__photon_flux_3D_polarization_y = \
                self.__extend_photon_flux_using_symmetries(
                    self.__photon_flux_3D_polarization_y)

    def __show_angular_distribution(self, z):
        fig = plt.figure(figsize=[12, 10])
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(1000*self.x_2D,
                               1000*self.y_2D,
                               z,
                               cmap=cm.coolwarm,
                               linewidth=0,
                               antialiased=False)
        ax.set_xlabel("\n"+r"$\theta_x$, mrad",
                      fontsize=self.label_font_size,
                      linespacing=self.label_linespacing)
        ax.set_ylabel("\n"+r"$\theta_y$, mrad",
                      fontsize=self.label_font_size,
                      linespacing=self.label_linespacing)
        return ax

    def __show_spectral_distribution(self, z):
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.plot(self.lambda_range, z)
        ax.set_xlabel(r"$\lambda$, um",
                      fontsize=self.label_font_size,)
        return ax

    def get_angular_distribution(self,
                                 polarization='sum',
                                 index_of_lambda=None):
        if index_of_lambda is not None:
            z = self.get_photon_flux_3D(polarization)[index_of_lambda]
        else:
            z = self.lambda_step *\
                np.apply_over_axes(
                    np.sum,
                    self.get_photon_flux_3D(polarization),
                    [0])[0]
        return z

    def get_spectral_distribution(self,
                                  polarization='sum',
                                  angular_indexes_tuple=None):
        if angular_indexes_tuple is not None:
            i, j = angular_indexes_tuple
            z = self.get_photon_flux_3D(polarization)[:, j, i]
        else:
            z = self.x_step*self.y_step \
                * np.apply_over_axes(
                    np.sum,
                    self.get_photon_flux_3D(polarization),
                    [1, 2]).reshape(-1)
        return z

    def show_angular_distribution(self,
                                  polarization='sum',
                                  index_of_lambda=None):
        z = self.get_angular_distribution(polarization, index_of_lambda)
        ax = self.__show_angular_distribution(z)
        if index_of_lambda is not None:
            dim = r" $\frac{\mathrm{Ph}}{\mathrm{rad}^2\mathrm{um}}$"
            ax.set_zlabel("\n"+r"$\frac{dN}{d\theta_x d\theta_y d\lambda}$,"
                          + dim,
                          fontsize=self.label_font_size,
                          linespacing=self.label_linespacing)
            lambda_val = self.lambda_range[index_of_lambda]
            ax.set_title(r"Polarization: "+polarization+", "
                         r"$\lambda$ = "
                         + "{:.3f} um".format(lambda_val))
            plt.show()
        else:
            dim = r" $\frac{\mathrm{Ph}}{\mathrm{rad}^2}$"
            ax.set_zlabel("\n"+r"$\frac{dN}{d\theta_x d\theta_y}$,"+dim,
                          fontsize=self.label_font_size,
                          linespacing=self.label_linespacing)
            ax.set_title("Polarization: "+polarization
                         + ", integrated over entire wavelength range")
            plt.show()

    def show_spectral_distribution(self,
                                   polarization='sum',
                                   angular_indexes_tuple=None):
        z = self.get_spectral_distribution(polarization, angular_indexes_tuple)
        ax = self.__show_spectral_distribution(z)
        if angular_indexes_tuple is not None:
            i, j = angular_indexes_tuple
            dim = r" $\frac{\mathrm{Ph}}{\mathrm{rad}^2\mathrm{um}}$"
            ax.set_ylabel(r"$\frac{dN}{d\theta_x d\theta_y d\lambda}$,"
                          + dim,
                          fontsize=self.label_font_size,
                          linespacing=self.label_linespacing)
            th_x = 1000*self.x_range[i]
            th_y = 1000*self.y_range[j]
            ax.set_title(r"Polarization: "+polarization+", "
                         r"$\theta_x$ = "
                         + "{:.3f} mrad, ".format(th_x)
                         + r"$\theta_y$ = "
                         + "{:.3f} mrad".format(th_y)
                         )
            plt.show()
        else:
            dim = r" $\frac{\mathrm{Ph}}{\mathrm{um}}$"
            ax.set_ylabel(r"$\frac{dN}{d\lambda}$,"+dim,
                          fontsize=self.label_font_size,
                          linespacing=self.label_linespacing)
            ax.set_title("Polarization: "+polarization
                         + ", integrated over entire angular range")
            plt.show()

    def get_total_photon_flux(self, polarization='sum'):
        z = self.get_spectral_distribution(polarization)
        return self.lambda_step*sum(z)
