
# Cindeldi/Sintef Database Cheat Sheet

This is a short summary of how we extracted the data from Skagerak's database.

The database lives within a docker container on a remote computer, and the database itself is a PostgreSQL.  
The general steps are:

1. Query the data from database and save it to a CSV-file.
2. Extract the file from docker container into remote computer.
3. Extract the file from remote to local computer.


The steps and commands are as follows.

1. Log into SSH - `ssh <USER>@129.241.64.67`.
2. Enter cineldi-container database - `docker exec -it cineldi_postgresql psql -U postgres`.
3. Connect to skagerak database - `\c skagerak`
4. Inside the database SQL-queries can be used, e.g. - `SELECT * FROM "MicroSCADA OPC DA.Skagerak_Microgrid.APL.1.P.NT1_RTU_N.1039" LIMIT 10;`
5. To copy out a file - `\copy "<signalname>" to filename.csv csv header`, e.g. `\copy "MicroSCADA OPC DA.Skagerak_Microgrid.APL.1.P.PV1_RTU_P_MET.1058" to GHI.csv csv header`.
6. Exit the datebase using `\q`.
7. Helpful command to view files in the docker container - `docker exec -t -i cineldi_postgresql  /bin/bash`.
8. Get file from docker-container to remote directory - `docker cp cineldi_postgresql:<path-to-file> ./`
9. Extract file from remote to local computer - Either use `scp remote_username@10.10.0.2:/remote/file.txt /local/directory` or [Filezilla](https://filezilla-project.org/)


