"""Independent exact-equation087 evaluator read decomposition, not full model."""
def split_slow(matrix,key,value,rho,eta):
    n=len(key)
    if len(matrix)!=n or len(value)!=n or any(len(row)!=n for row in matrix):raise ValueError('square matched memory')
    mk=[sum(matrix[i][j]*key[j] for j in range(n)) for i in range(n)]
    historical=[[rho*matrix[i][j]-eta*rho*mk[i]*key[j] for j in range(n)] for i in range(n)]
    current=[[eta*value[i]*key[j] for j in range(n)] for i in range(n)]
    return historical,current

def attributed_read(matrix,key,value,query,rho,eta,mode):
    historical,current=split_slow(matrix,key,value,rho,eta)
    n=len(key)
    if mode=='old_zero':used=current
    elif mode=='current_zero':used=historical
    elif mode=='full':used=[[historical[i][j]+current[i][j] for j in range(n)] for i in range(n)]
    elif mode=='old_permuted':
        permuted=[matrix[(i-1)%n][:] for i in range(n)]
        old,_=split_slow(permuted,key,value,rho,eta)
        used=[[old[i][j]+current[i][j] for j in range(n)] for i in range(n)]
    else:raise ValueError('registered attribution')
    return [sum(used[i][j]*query[j] for j in range(n)) for i in range(n)]
