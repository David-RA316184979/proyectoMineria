<?php include("Funciones.php"); ?>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>

<meta charset="UTF-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">

<meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>Consultar datos</title>

<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">

<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Bree+Serif&display=swap" rel="stylesheet">

<style type="text/css">
<!--
body {
	background-image: url();
	background-repeat: no-repeat;
	background-color: #333333;
}
-->
</style>
	</head>
		<style> 
	#target { 
		font: Verdana, Geneva, sans-serif; 	
		color: black;
	    text-align:center;			
	} 	
	#targeta { 
		font: Verdana, Geneva, sans-serif; 
		color: black; 
		text-align:justify;
	} 	
	.dib1 {  
		position: absolute; left: 1060px; top: 35px; width: 65px; height: 45px; 
	}
	.nav-link{
      font-size: 20px;
      font-family: 'Bebas Neue';
    }
	#global {
      max-width: 100%;
      height: auto;
      overflow-x: scroll;
    }
    #mensajes {
      height: auto;
    }
	a {
		text-decoration: none;
		color: white;
	} 
</style> 

<body style="background-color: #f1ece7;">

&nbsp;

<div width="50%">
<table class="table table-sm table-hover table-responsive">  
	<tr>
		<th id="target"></th>
    	<th id="target"></th>
        <th id="target"></th>
		<th id="target"></th>
		<th id="target"></th>
        <th id="target"></th>
	</tr>

	<?php
		$resultado_data=oci_parse($conexion,$sql_data);
		oci_execute($resultado_data);
		while( $fila_data = oci_fetch_assoc($resultado_data))  // OCI_BOTH OCI_NUM  OCI_ASSOC OCI_RETURN_NULLS   OCI_ASSOC+OCI_RETURN_NULLS  OCI_RETURN_LOBS
		{
	?>

	<tr  id="target">
		<td><?php echo $fila_data['ID_data']; ?></td>	
    	<td><?php echo $fila_data['']; ?></td>	
        <td><?php echo $fila_data[""]; ?></td>
		<td><?php echo $fila_data[""]; ?></td>
		<td><?php echo $fila_data[""]; ?></td>
        <td><?php echo $fila_data[""]; ?></td>
		

		<td> 
		<button type="button" class="btn btn-success">
		<?php echo "<center><a href='Funciones.php?id=".$fila_data['ID_data']."'>Editar</a></h1></center>";
		?>	
		</button>
		</td>

		<td>
		<button type="button" class="btn btn-danger">
		<?php echo "<center><a href='Funciones.php?id=".$fila_data['ID_data']."'>Eliminar</a></h1></center>";
		?>	
		</button> 
		</td>
				
		</td>
    </tr>	
<?php
	}
?>
</div>

</table>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-MrcW6ZMFYlzcLA8Nl+NtUVF0sA7MsXsP1UyJoMp4YLEuNSfAP+JcXn/tWtIaxVXM" crossorigin="anonymous"></script>
</body>
</html>
