from main.assignment_2 import nevilles_interpolation, newtons_forward_method, newtons_forward_interpolation, hermite_interpolation, cubic_spline_interpolation

class Testing:
    def __init__(self):
        pass

    def run_tests(self):
        self.test_nevilles_interpolation()
        self.test_newtons_forward_method()
        self.test_hermite_interpolation()
        self.test_cubic_spline_interpolation()

    def test_nevilles_interpolation(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        x_target = 3.7
        result = nevilles_interpolation(x_vals, y_vals, x_target)
        print("Neville's Method, Approximation of f(3.7):")
        print(f"{result:.16f}")
    
    def test_newtons_forward_method(self):
        x_vals = [7.2, 7.4, 7.5, 7.6]
        y_vals = [23.5492, 25.3913, 26.8224, 27.4589]
        newton_table = newtons_forward_method(x_vals, y_vals)
        newton_coeffs = [newton_table[0, i] for i in range(4)]
    
        print("\nNewton's Forward Coefficients:")
        for coeff in newton_coeffs[1:4]:
            print(f"{coeff:.15f}")  
    
        x_interp = 7.3
        result = newtons_forward_interpolation(x_vals, newton_coeffs, x_interp)
        print(f"\nf({x_interp}) ≈ {result:.15f}")
        
    
    def test_hermite_interpolation(self):
        x_vals = [3.6, 3.8, 3.9]
        y_vals = [1.675, 1.436, 1.318]
        dy_vals = [-1.195, -1.188, -1.182]

        z, Q = hermite_interpolation(x_vals, y_vals, dy_vals)

        print("\nHermite Interpolation Table:")
    
        for i in range(len(z)):
            row = [z[i]] + [Q[i][j] for j in range(2*len(x_vals)-2)]
            print("[ " + " ".join(f"{val: .8e}" for val in row) + " ]")


    def test_cubic_spline_interpolation(self):
        x_vals = [2, 5, 8, 10]
        y_vals = [3, 5, 7, 9]
        coef_matrix, rhs_vector, spline_coeffs = cubic_spline_interpolation(x_vals, y_vals)
    
        print("\nCubic Spline Matrix:")
        print(coef_matrix)
        print(rhs_vector)
        print(spline_coeffs)
        

if __name__ == "__main__":
     Testing().run_tests()