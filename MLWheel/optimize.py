#!/usr/bin/env
# -*- coding: utf-8 -*-

'''

optimize method

1. import "optimize" from "MLWheel" module

	>>> from MLWheel import optimize

2. method

	(1) Lagrange Multiplier

		>>> optimize.lagrange_multiplier(obj, cons_e, cons_le, *var)

		obj: objective function

		cons_e: list for equal constraints

		cons_le: list for less equal constraints

		var: original variebles

		!!! the results include complex number !!!


'''

__author__ = 'YayoYY'

from sympy import *
from functools import reduce

def lagrange_multiplier(obj, cons_e, cons_le, *var):

	# equal funcions

	lagrange_function = obj
	functions = []
	multipliers_e = []
	multipliers_le = []

	for i, item in enumerate(cons_e):
		multiplier = Symbol(chr(i+65))
		multipliers_e.append(multiplier)
		functions.append(item)
		lagrange_function = lagrange_function + multiplier * item

	for i, item in enumerate(cons_le):
		multiplier = Symbol(chr(i+97))
		multipliers_le.append(multiplier)
		functions.append(multiplier * item)
		lagrange_function = lagrange_function + multiplier * item

	for item in var:
		functions.append(diff(lagrange_function, item))

	symbols = list(var) + multipliers_e + multipliers_le

	raw_results = solve(functions, symbols)

	# non-equal funcions

	var_len = len(var)
	e_len = len(multipliers_e)
	le_len = len(multipliers_le)

	results = []

	for item in raw_results:
		e = [x for x in item[var_len:var_len + e_len] if x == 0]
		le = [x for x in item[var_len + e_len:] if x < 0]
		g = [ x for x in [reduce(sub_reduce, [g] + [(x,y) for x,y in zip(symbols, item)]) for g in cons_le] if x > 0]

		if e or le or g:
			break
		else:
			results.append(item)

	# solve optimize

	losses = [reduce(sub_reduce, [obj] + [(x,y) for x,y in zip(symbols, results[i])]) for i in range(len(results))]
	min_loss = min(losses)
	optimize_result = [(x,y) for x,y in zip(results,losses) if y == min_loss]

	return optimize_result

def sub_reduce(function, rep):
	symbol = rep[0]
	value = rep[1]
	return function.subs(symbol, value)