import matplotlib.pyplot as plt
import numpy as np
from fenics import *
from guccionematerial import *


def load_simplified_data():
    """
    Versão simplificada que carrega apenas os dados de geometria que você tem.
    Retorna 3 itens: a malha, os marcadores de contorno e a numeração.
    """
    print("Carregando malha de 'Patient_7.xml'...")
    mesh = Mesh(MPI.comm_world, "data/discretizacao_02/Patient_10/Patient_10.xml")
    
    print("Carregando marcadores de contorno de 'facet_region.xml'...")
    mf = MeshFunction("size_t", mesh, "data/discretizacao_02/Patient_10/facet_region.xml")

    numbering = {
        "BASE": 10,
        "ENDO": 30,
        "EPI": 40
    }

    # Retira fibras
    
    return mesh, mf, numbering


def compute_cavity_volume(mesh, mf, numbering, u=None):
    """
    Esta função calcula o volume da cavidade.
    Ela permanece exatamente igual ao original.
    """
    X = SpatialCoordinate(mesh)
    N = FacetNormal(mesh)

    if u is not None:
        I = Identity(3) # a matriz identidade
        F = I + grad(u) # o gradiente de deformação
        J = det(F)
        vol_form = (-1.0/3.0) * dot(X + u, J * inv(F).T * N)
    else:
        # Se não houver deformação (u=None), calcula o volume da geometria inicial
        vol_form = (-1.0/3.0) * dot(X, N)

    ds = Measure('ds', domain=mesh, subdomain_data=mf)

    return assemble(vol_form * ds(numbering["ENDO"]))

#print("Iniciando script...")

mesh, boundary_markers, numbering = load_simplified_data()
print("Dados de geometria carregados com sucesso.")

# --- Calculando os volumes ---

# Para obter a fórmula do volume (a mesma para ambos os cálculos)
X = SpatialCoordinate(mesh)
N = FacetNormal(mesh)
vol_form = (-1.0/3.0) * dot(X, N)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# 1. Calcular o volume da CAVIDADE (usando a superfície do ENDOcárdio)
volume_cavidade = assemble(vol_form * ds(numbering["ENDO"]))

# 2. Calcular o volume TOTAL DO MÚSCULO (usando a superfície do EPIcárdio)
volume_total = assemble(vol_form * ds(numbering["EPI"]))

print("-" * 40)
print(f"Volume da Cavidade (ENDO): {volume_cavidade:.2f}")
print(f"Volume Total do Ventrículo (EPI): {volume_total:.2f}")
print("-" * 40)