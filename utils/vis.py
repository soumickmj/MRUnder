def getMaxSubSquare(M):
    R = len(M) # no. of rows in M[][]
    C = len(M[0]) # no. of columns in M[][]
 
    S = []
    for i in range(R):
      temp = []
      for j in range(C):
        if i==0 or j==0:
          temp += M[i][j],
        else:
          temp += 0,
      S += temp,
    # here we have set the first row and first column of S same as input matrix, other entries are set to 0
 
    # Update other entries
    for i in range(1, R):
        for j in range(1, C):
            if (M[i][j] == 1):
                S[i][j] = min(S[i][j-1], S[i-1][j],
                            S[i-1][j-1]) + 1
            else:
                S[i][j] = 0
     
    # Find the maximum entry and
    # indices of maximum entry in S[][]
    max_of_s = S[0][0]
    max_i = 0
    max_j = 0
    for i in range(R):
        for j in range(C):
            if (max_of_s < S[i][j]):
                max_of_s = S[i][j]
                max_i = i
                max_j = j
    print(max_of_s)
    print(((max_of_s*max_of_s)/M.size) *100)