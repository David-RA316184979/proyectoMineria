<?php
// Conexion con Oracle

$conexion = oci_connect("gblizarragar","bet1234","localhost/OLTP");

if ($conexion) {
echo "Conexion exitosa";  // Mensaje Conectado a Oracle
}
else {
   $m = oci_error();
   echo $m['message'], "\n";
}

?>
