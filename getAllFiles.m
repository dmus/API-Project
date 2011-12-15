function fileList = getAllFiles(dirName)
%GETALLFILES lists all files in given directory and subdirectory
  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  
  if ~isempty(fileList)
    fileList = fileList(find(~cellfun(@isO,fileList))); 
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end

end

function  a = isO(filename)
    a = 0;
    rev = fliplr(filename);
    label = rev(12);
    if strcmp(label, 'O')
        a = 1;
    end
end