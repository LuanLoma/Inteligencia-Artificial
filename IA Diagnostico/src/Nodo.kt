internal class Nodo(var nombre: String) {
    var izquierda: Nodo?
    var derecha: Nodo? = null

    init {
        this.izquierda = this.derecha
    }
}