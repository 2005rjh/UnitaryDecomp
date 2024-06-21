import numpy as np
#from fractions import Fraction as F



def joinstr(strlist, valign='c', delim=''):
	# TODO, does not handle tabs properly
	# TODO, vary the vertical justification
	""" Join strings with multilines
		no newline at the end of everything

		with tabs, it tries its best to guess where it is (if any of the strings has more than one line)
		"""
	## Below are all lists, an item for each one in strlist
	numstr = len(strlist)
	listlines = []		# list of string in strlist[i]
	numlines = []	# number of lines in strlist[i]
	#strwidth = []		# max width of strlist[i]
	empty_str = []		# a string with only spaces and tabs, with strwidth number of characters (use for padding)
	## Parse and process input data
	for s in strlist:
		if isinstance(s, str):
			list_of_lines = s.split('\n')
		else:
			list_of_lines = str(s).split('\n')		# convert to string
		listlines.append(list_of_lines)
		numlines.append(len(list_of_lines))
		list_of_str_lengths = list(len(l) for l in list_of_lines)
		the_longest_line = list_of_lines[np.argmax(list_of_str_lengths)]
		empty_maxlen_liststr = [' '] * len(the_longest_line)
		for i in range(len(the_longest_line)):
			if the_longest_line[i] == '\t': empty_maxlen_liststr[i] = '\t'
		empty_str.append( "".join(empty_maxlen_liststr) )
	maxlines = max(numlines)
	s = ""
	for i in range(maxlines):
		for t in range(numstr):
			if i < int((maxlines - numlines[t]) / 2)  or  i >= int((maxlines - numlines[t]) / 2) + numlines[t]:
				s += empty_str[t]
			else:
				print_str = listlines[t][i - int((maxlines - numlines[t]) / 2)]
				s += print_str + empty_str[t][len(print_str):]
			if t < numstr - 1: s += delim
		if i < maxlines - 1: s += '\n'
	return s




def to_mathematica_lists(a):
	""" curly brackets """
#	if not isinstance(a, np.ndarray): raise ValueError
	if isinstance(a, str): return '"' + str(a) + '"'
	try:
		iter(a)
		s = "{"
		for i,suba in enumerate(a):
			if i > 0: s += ", "
			s += to_mathematica_lists(suba)
		s += "}"
		return s
	except TypeError:
		if isinstance(a, float) or isinstance(a, complex):
			return str(a).replace('e', '*^').replace('j', ' I').replace('inf', '\[Infinity]')
		return str(a)


def matrixstr(strlistlist, delim = ' ', linehead = '[', linetail = ']', **kwargs):
	"""Format a table (2D array) of objects into a human-readable string.
	Parameters:
		strlistlist:  a 2-array of strings.
		delim:
		linehead:
		linetail:
	More Parameters:
		sameColWidth, lines_between_rows.
	Returns:
		A string (with no newline at the end)
	"""
##TODO, assumes all strings have one line.
	blankpad = ''		# this fills the missing elements at the end of each row (to make # of columns match)

	nrow = len(strlistlist)
	if nrow == 0: return ''
	numcol = [ len(sl) for sl in strlistlist ]
	ncol = max(numcol)
	strtable = []
	strwidthtable = np.zeros((nrow, ncol), int)
	for i in range(nrow):
		strl = list(map(str, strlistlist[i]))
		if len(strl) < ncol: strl = strl + [blankpad] * (ncol - len(strl))
		strwidth = list(map(len, strl))
		strtable.append(strl)
		strwidthtable[i] = np.array(strwidth)
	colwidths = np.amax(strwidthtable, axis=0)
	if kwargs.get('sameColWidth', False): colwidths.fill( np.max(colwidths) )
	delim_len = len(delim)
	total_internal_str_width = np.sum(colwidths) + (ncol-1) * delim_len

	if kwargs.get('lines_between_rows', 0) > 0: str_between_lines = linehead + ' '*total_internal_str_width + linetail + '\n'
	halgn = kwargs.get('halign', 'c')

	s = ''
	for i in range(nrow):
		if i > 0:
			s += '\n'
			if kwargs.get('lines_between_rows', 0) > 0: s += str_between_lines * kwargs['lines_between_rows']
		s += linehead
		for j in range(ncol):
			if j > 0: s += delim
			pad = colwidths[j] - len(strtable[i][j])
			if halgn == 'l': s += strtable[i][j] + ' ' * pad
			if halgn == 'c' or halgn == 'cl': s += ' ' * (pad//2) + strtable[i][j] + ' ' * (pad-pad//2)
			if halgn == 'cr': s += ' ' * (pad-pad//2) + strtable[i][j] + ' ' * (pad//2)
			if halgn == 'r': s += ' ' * pad + strtable[i][j]
		s += linetail
	return s



def LaTeX_frac(f):
	"""Convert a fraction to latex string.
	f is a fraction (python builtin from class fractions),
	returns a string."""
	if f.denominator == 1: return str(f)
	s = "\\frac{{{}}}{{{}}}".format(abs(f.numerator), f.denominator)
	if f < 0: s = "-" + s
	return s
	
