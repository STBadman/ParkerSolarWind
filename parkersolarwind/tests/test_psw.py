import parkersolarwind as psw
import astropy.units as u
import astropy.constants as const
import numpy as np
import pytest

def fext_success(r) : 
    '''
    To work correctly, a force function must take an array of distances
    and return an array of the same shape with units equivalent to
    GM_sun/R_sun^2
    '''
    return (const.G*const.M_sun/const.R_sun**2)*np.ones(len(np.array(r)))

def fext_fail(r) : return None

def ifext_success(r1,r2) : 
    '''
    To work correctly, an integrated force function must take an 
    array of inner and outer distances (integration limits) and 
    return an array of floats the same shape with units of (km/s)^2 
    (i.e. output should not be astropy.units.quantity.Quantity)
    '''
    return (const.G*const.M_sun/const.R_sun).to("km^2/s^2").value*np.ones(len(np.array(r2)))

def ifext_fail(r1,r2) : return None

def test_coronal_base_altitude() :
    # Check default output is a distance
    assert psw.coronal_base_altitude().unit.is_equivalent(u.m), "must return a distance"

def test_critical_speed() :
    # Check output is a velocity
    assert psw.critical_speed().unit.is_equivalent(u.m/u.s), "must return a velocity"

def test_critical_radius() :
    # Check output is a distance
    assert psw.critical_radius().unit.is_equivalent(u.m), "must_return a distance"

def test_critical_radius_fext(fext_success,fext_fail) :
    # Check output is a distance
    # Check error handling for fext function

    assert False

def test_ug() :
    # Check output is a velocity
    assert psw.get_ug(1*u.R_sun).unit.is_equivalent(u.m/u.s), "must return velocity"

def test_uc0() : 
    # Check output is a velocity
    assert psw.get_uc0(5/3,1*u.MK).unit.is_equivalent(u.m/u.s), "must return velocity"

def test_scrit() :
    # Check output is dimensionless
    assert type(psw.s_crit(1,1*u.MK,5/3)) is not u.quantity.Quantity, "must return dimensionless"
    # check output shape matches input shape
    array_test = np.ones(10)
    assert psw.s_crit(array_test,1*u.MK,5/3).shape == array_test.shape, "output shape must match input shape"

def test_scrit_fext() :
    array_test = np.ones(10)
    out = psw.s_crit_fext(array_test,1*u.MK,5/3,
                          fext_success,ifext_success
                        )
    # Check output is dimensionless
    assert out.unit.is_equivalent(u.m/u.m), "must return dimensionless"
    # check output shape matches input shape
    assert out.shape == array_test.shape
    # Check error handling for fext function
    with pytest.raises(AssertionError) :
        psw.s_crit_fext(array_test,1*u.MK,5/3,[],[])

def test_scrit_solution() :
    # Check output is 2-array, dimensionless
    # Check error handling for fext function
    assert False

def test_scrit_fext_solution() :
    # Check output is a 2-array, dimensionless
    # Check error handling for fext function
    assert False

def test_u0() :
    # Check output is velocity
    assert False

def test_u0_fext() :
    # Check output is velocity
    # Check error handling for fext function
    assert False

def test_uc_polytropic() :
    # Check output is velocity
    assert False

def test_uc_polytropic_fext() :
    # Check output is velocity
    # Check error handling for fext function
    assert False

def test_parker_isothermal_algebraic() :
    # Check output is dimensionless
    assert False

def test_parker_isothermal_fext_algebraic() :
    # Check output is dimensionless
    # Check fext error handling
    assert False

def test_parker_polytropic_algebraic() :
    # Check output is (km/s)^2 or equivalent
    assert False

def test_parker_polytropic_fext_algebraic() :
    # Check output is (km/s)^2 or equivalent
    # Check fext error handling
    assert False

def test_isothermal_solution() :
    # Check input error handling
    # Check outputs 4 items
    # Check units of outputs
    assert False

def test_isothermal_fext_solution() :
    # Check input error handling including fext and ifext
    # Check outputs 4 items
    # Check units of outputs
    assert False

def test_polytropic_solution() :
    # Check input error handling
    # Check outputs 9 items
    # Check units of outputs
    assert False

def test_polytropic_fext_solution() :
    # Check input error handling including fext and ifext
    # Check outputs 9 items
    # Check units of outputs
    assert False

def test_isothermal_layer_solution() : 
    # Check input error handling
    # Check outputs 10 items
    # Check units of outputs
    assert False

def test_isothermal_layer_fext_solution() : 
    # Check input error handling including fext and ifext
    # Check outputs 10 items
    # Check units of outputs
    assert False