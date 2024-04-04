import cx from 'clsx'
import { useState } from 'react'
import { Table, ScrollArea, ActionIcon, Text, Group } from '@mantine/core'
import classes from './FileList.module.css'
import { IconExternalLink, IconX } from '@tabler/icons-react'
import { FilePreviewModal } from './FilePreviewModal.tsx'
import { FileWithId } from '../App.tsx'

export interface FileListProps {
  files: FileWithId[]
  setFiles: any
}

export const FileList = ({ files, setFiles }: FileListProps) => {
  const [scrolled, setScrolled] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewModalOpen, setPreviewModalOpen] = useState(false)

  const handleRemoveFile = (idToRemove: string) => {
    setFiles((prevFiles: FileWithId[]) =>
      prevFiles.filter((file: FileWithId) => file.id !== idToRemove)
    )
  }

  const handleFileClick = (file: File) => {
    setSelectedFile(file)
    setPreviewModalOpen(true)
  }

  const filesExpand = files.map((file: FileWithId) => (
    <Table.Tr key={file.id}>
      <Table.Td className={classes.tdName}>
        <Group>
          <Text>{file.file.name}</Text>
          <ActionIcon
            onClick={() => handleFileClick(file.file)}
            variant="light"
          >
            <IconExternalLink />
          </ActionIcon>
        </Group>
      </Table.Td>
      <Table.Td>
        <ActionIcon
          color="red"
          variant="filled"
          onClick={() => handleRemoveFile(file.id)}
        >
          <IconX />
        </ActionIcon>
      </Table.Td>
    </Table.Tr>
  ))

  return (
    <>
      <ScrollArea
        onScrollPositionChange={({ y }) => setScrolled(y !== 0)}
        className={cx(classes.table)}
      >
        <Table>
          <Table.Thead
            className={`${cx(classes.header, { [classes.scrolled]: scrolled })}`}
          >
            <Table.Tr>
              <Table.Th className={classes.thName}>Name</Table.Th>
              <Table.Th className={classes.thRemove}>Remove</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>{filesExpand}</Table.Tbody>
        </Table>
      </ScrollArea>
      {selectedFile && (
        <FilePreviewModal
          file={selectedFile}
          isModalOpen={previewModalOpen}
          setIsModalOpen={setPreviewModalOpen}
        />
      )}
    </>
  )
}
