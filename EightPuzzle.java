import java.util.*;

public class EightPuzzle {
    private static final int[][] GOAL = {{1, 2, 0}, {3, 4, 5}, {7, 8, 6}};
    private static final int MAX_MOVIMIENTOS = 1020;  // Límite de movimientos para DFS

    public static void main(String[] args) {
        int[][] initial = {
                {1, 2, 3},
                {4, 0, 5},
                {7, 8, 6}
        };

        System.out.println("Resolviendo con BFS:");
        solveWithBFS(initial);

        System.out.println("\nResolviendo con DFS limitado a " +(MAX_MOVIMIENTOS)+ " movimientos:");
        solveWithDFS(initial);

        System.out.println("\nResolviendo con A*:");
        solveWithAStar(initial);
    }

    static void solveWithBFS(int[][] initial) {
        Queue<Node> cola = new LinkedList<>();
        Set<String> visitados = new HashSet<>();
        Node nodoInicial = new Node(initial, null, 0);
        cola.add(nodoInicial);
        visitados.add(nodoInicial.getEstado());

        while (!cola.isEmpty()) {
            Node actual = cola.poll();

            if (actual.esMeta()) {
                imprimirSolucion(actual);
                return;
            }

            for (Node siguiente : actual.generarHijos()) {
                String estado = siguiente.getEstado();
                if (!visitados.contains(estado)) {
                    visitados.add(estado);
                    cola.add(siguiente);
                }
            }
        }
        System.out.println("No se encontró solución.");
    }

    static void solveWithDFS(int[][] initial) {
        Set<String> visitados = new HashSet<>();
        Node nodoInicial = new Node(initial, null, 0);

        if (!dfsLimited(nodoInicial, visitados)) {
            System.out.println("No se encontró solución dentro del límite de movimientos.");
        }
    }

    static boolean dfsLimited(Node nodo, Set<String> visitados) {
        if (nodo.esMeta()) {
            imprimirSolucion(nodo);
            return true;
        }

        if (nodo.movimientos > MAX_MOVIMIENTOS) {
            return false;  // Si supera el límite de movimientos, abortamos
        }

        visitados.add(nodo.getEstado());
        for (Node siguiente : nodo.generarHijos()) {
            if (!visitados.contains(siguiente.getEstado())) {
                if (dfsLimited(siguiente, visitados)) {
                    return true;
                }
            }
        }
        return false;
    }

    static void solveWithAStar(int[][] initial) {
        PriorityQueue<Node> colaPrioridad = new PriorityQueue<>(Comparator.comparingInt(n -> n.prioridad));
        Set<String> visitados = new HashSet<>();
        Node nodoInicial = new Node(initial, null, 0);
        nodoInicial.prioridad = nodoInicial.distanciaManhattan() + nodoInicial.movimientos;
        colaPrioridad.add(nodoInicial);

        while (!colaPrioridad.isEmpty()) {
            Node actual = colaPrioridad.poll();

            if (actual.esMeta()) {
                imprimirSolucion(actual);
                return;
            }

            if (visitados.contains(actual.getEstado())) continue;
            visitados.add(actual.getEstado());

            for (Node siguiente : actual.generarHijos()) {
                siguiente.prioridad = siguiente.distanciaManhattan() + siguiente.movimientos;
                colaPrioridad.add(siguiente);
            }
        }
        System.out.println("No se encontró solución.");
    }

    static void imprimirSolucion(Node nodo) {
        List<Node> camino = new ArrayList<>();
        while (nodo != null) {
            camino.add(nodo);
            nodo = nodo.padre;
        }
        Collections.reverse(camino);

        System.out.println("Solución encontrada en " + (camino.size() - 1) + " movimientos:");
        for (Node n : camino) {
            imprimirTablero(n.tablero);
            System.out.println();
        }
    }

    static void imprimirTablero(int[][] tablero) {
        for (int[] fila : tablero) {
            for (int num : fila) {
                System.out.print(num + " ");
            }
            System.out.println();
        }
    }

    static class Node {
        int[][] tablero;
        Node padre;
        int movimientos;
        int prioridad;

        public Node(int[][] tablero, Node padre, int movimientos) {
            this.tablero = new int[3][3];
            for (int i = 0; i < 3; i++) {
                this.tablero[i] = Arrays.copyOf(tablero[i], 3);
            }
            this.padre = padre;
            this.movimientos = movimientos;
        }

        String getEstado() {
            StringBuilder sb = new StringBuilder();
            for (int[] fila : tablero) {
                for (int num : fila) {
                    sb.append(num).append(",");
                }
            }
            return sb.toString();
        }

        boolean esMeta() {
            return Arrays.deepEquals(tablero, GOAL);
        }

        List<Node> generarHijos() {
            List<Node> hijos = new ArrayList<>();
            int x = -1, y = -1;

            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    if (tablero[i][j] == 0) {
                        x = i;
                        y = j;
                        break;
                    }
                }
                if (x != -1) break;
            }

            int[][] direcciones = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (int[] dir : direcciones) {
                int nuevoX = x + dir[0];
                int nuevoY = y + dir[1];

                if (nuevoX >= 0 && nuevoX < 3 && nuevoY >= 0 && nuevoY < 3) {
                    int[][] nuevoTablero = new int[3][3];
                    for (int i = 0; i < 3; i++) {
                        nuevoTablero[i] = Arrays.copyOf(tablero[i], 3);
                    }
                    nuevoTablero[x][y] = nuevoTablero[nuevoX][nuevoY];
                    nuevoTablero[nuevoX][nuevoY] = 0;
                    hijos.add(new Node(nuevoTablero, this, movimientos + 1));
                }
            }
            return hijos;
        }

        int distanciaManhattan() {
            int distancia = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    int valor = tablero[i][j];
                    if (valor != 0) {
                        int xMeta = (valor - 1) / 3;
                        int yMeta = (valor - 1) % 3;
                        distancia += Math.abs(i - xMeta) + Math.abs(j - yMeta);
                    }
                }
            }
            return distancia;
        }
    }
}